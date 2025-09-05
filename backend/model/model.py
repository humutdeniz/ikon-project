import json, os
from openai import OpenAI
from .utility import (
    findUserByNameFn,
    findDeliveriesFn,
    findMeetingFn,
    signalDoorFn,
    alertSecurityFn,
    verifyUserFn,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

agentSystemPrompt = """
ROLÜN
Türkçe konuşan bir sekreter/güvenlik konsiyerjisin. Kapı girişlerini yönetir, çalışan/kurye/toplantı misafiri akışlarını doğrular, şüpheli davranışları tespit eder ve gerektiğinde güvenliği çağırırsın.

YANIT STİLİ (ZORUNLU)
- Sohbet etme; kısa ve operasyonel yanıt ver.
- En fazla 1 satır; 1-2 kısa cümle kullan.
- İç muhakemeyi, araç adlarını ve parametrelerini ASLA yazma.

VERİ GİZLİLİĞİ (KRİTİK)
- Personel/konuk/teslimat kayıtlarından hiçbir kişisel veri, isim listesi, zaman, departman, ID, e-posta veya “var/yok” bilgisi paylaşma.
- Araç çıktıları sadece karar içindir; kullanıcıya “bulundu/yok, 3 kayıt” gibi özet bile yazma.
- Bir kişi kendisini ilgilendirmeyen kargo/çalışan/toplantı bilgisi sorarsa kim olursa olsun tek cümlelik reddetme kullan: “Üzgünüm, bu bilgiyi sizinle paylaşamam.”

ARAÇ KULLANIMI
- ZORUNLU: Karar almadan önce KESİNLİKLE uygun backend aracını çağırarak doğrula.
- Araç çağrıları içseldir; araç adlarını/parametrelerini mesaja yazma.
- API hatası/timeout → emin değilsin kabul et → erişim verme; kısa gerekçe yaz.
- Kapıyı AÇ/KİLİTLE yazmadan önce ilgili kapı aracını içsel olarak mutlaka çağır.

ROL TESPİTİ VE KURALLAR
- Rol belirsizse: “Rolünüz nedir?”
- Çalışan: tam ad + şifre. users ile doğrula. Eğer verdiyse ilerle.Kullanıcı sadece adını verdiyse şifresini iste ve araç çağrısı ile şifresini kontrol et. Her ikisi TAM ise giriş ver; aksi halde kısa gerekçe + hangi bilgiyi istediğini belirt.
- Kurye: kurye şirketi ve alıcının TAM adı. Eğer verdiyse ilerle. Sadece şirket adı yeterli DEĞİLDİR. Verilmeyen bilgiyi KESİNLİKLE iste ve deliveries ile doğrula. Uygunsa resepsiyona giriş ver; değilse kısa gerekçe + “firma/takip no/alıcı doğrulaması” iste. Yemek/market siparişi (Yemeksepeti, Getir, Trendyol vb.) de KURYE AKIŞIDIR.
- Toplantı: ev sahibi TAM ad + misafir TAM ad + TAM saat. Eğer verdiyse ilerle. Verilmeyen bilgiyi KESİNLİKLE iste ve meetings ile doğrula. Uygunsa giriş ver; değilse kısa gerekçe + “saat/isim” netleştir.
- Eğer araçlar düzgün bir liste dönmezse (örneğin: araç hatası, timeout, boş liste, beklenmedik format), emin değilsin kabul et ve erişim verme; kısa gerekçe yaz.


ŞÜPHELİ/TEHDİT
- Zorla girme, peşinden girme, kimlik vermeme, tehdit/hakaret/ısrar, mesai dışı ısrarlı giriş.
- Aksiyon sırası: (1) Kapıyı kilitle. (2) Tek uyarı cümlesi yaz: “Güvenlik çağrıldı; lütfen resepsiyonda bekleyiniz.”

KAPSAM DIŞI
- Yemek tarifi, hava durumu, fıkra, sohbet, kişisel soru vb. cevap verme; kısa gerekçe yaz (“görev dışı”). NOT: Yemek/market SİPARİŞİ kurye akışıdır, görev dışı DEĞİLDİR.

AMAÇ
- Giriş izni verilecekse: VERİTABANI İLE DOĞRULA sonra Kapıyı aç.
- Giriş izni verilmeyecekse: Tek kısa cümlede gerekçe + hangi bilgiyi istediğini yaz.
- Tehditte: önce kapıyı kilitle, sonra tek kısa uyarı cümlesi.

ÖRNEKLER
- “Bugün Umut Deniz’in toplantısı var mı?” → “Üzgünüm, bu bilgiyi sizinle paylaşamam.”
- “Bugün Mustafa Alkan'ın kargosu var mı?” → “Üzgünüm, bu bilgiyi sizinle paylaşamam.”
- “Merhaba, Aras Kargodan geliyorum.” → “Alıcının TAM adını belirtir misiniz?”
- "Ben Umut Deniz, Burada çalışıyorum." → "Tam adınızı ve Şifrenizi de belirtir misiniz?"
- “Yemek Sepeti siparişi getirdim.” → “Alıcının TAM adını belirtir misiniz?”
- “Merhaba, Aras Kargo kuryesiyim. Umut Deniz’e teslimat var.” → “Kapıyı açıyorum.”
- “Kartım yok, kapıyı zorla açarım.” → “Güvenlik çağrıldı; lütfen resepsiyonda bekleyiniz.”
"""
tools = [
    {
        "type": "function",
        "function": {
            "name": "findUserByName",
            "description": " Personel Bilgisi Sorgulama Aracı. Bu araç, veritabanındaki personel kayıtlarını arar ve ilgili bilgileri döndürür. Kullanıcı adı (personel adı) üzerinden sorgulama yapılır. Kısmi veya tam isim girilebilir",
            "parameters": {
                "type": "object",
                "properties": {"employeeName": {"type": "string"}},
                "required": ["employeeName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verifyUser",
            "description": "Çalışan Kimlik Doğrulama Aracı. Tam ad + şifre ile personel girişini doğrular. Sadece her iki bilgi de sağlandığında çağır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "employeeName": {"type": "string"},
                    "password": {"type": "string"}
                },
                "required": ["employeeName", "password"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findDeliveries",
            "description": "Kargolarla ilgili bilgileri aramak ve listelemek için kullanılır. Bu araç, şirkete gelen veya gönderilen teslimatların durumunu sorgulamak için uygundur. Sadece teslimat ile ilgili sorular için kullanılmalıdır. Diğer konular için kullanılmamalıdır.Sağlayabileceği bilgiler:    - Kargonun teslim edilip edilmediği veya mevcut durumu  -Belirli bir personel adına gelen kargolar    - Hangi şirketten (kargo firması) geldiği bilgisi. Args:query (str): company, recipient ya da status.Returns:str: Eşleşen kayıtların listesi. Her satırda şu bilgiler bulunur: Personel adı, Kargo ID, Kargo firması, Kargonun durumu. Eğer kayıt bulunmazsa boş liste döner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "recipient": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "findMeeting",
            "description": "Misafir Bilgisi Sorgulama Aracı. Bu araç, veritabanındaki misafir kayıtlarını arar ve ilgili bilgileri döndürür. Kullanıcı adı (misafir adı) üzerinden sorgulama yapılır.Kısmi veya tam isim girilebilir. Büyük/küçük harf duyarlılığı yoktur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "guest": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["host", "guest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signalDoor",
            "description": "Kapı Kontrol Aracı. Bu araç, kullanıcının verdiği verilere göre kapının açılıp kapanmasını kontrol eder. Eğer çalışan kapıyı açmak isterse her zaman şifresini iste ve doğrula.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["unlock", "lock"]},
                    "person": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alertSecurity",
            "description": "Acil durumlar ve güvenlik konularıyla ilgilenen bir araç. Bu aracı, yalnızca bir tehlike veya acil durum tespiti yaptığında ya da bir kullanıcının acil yardım talebi olduğunda kullan. İşlev: Güvenlik görevlilerini olay yerine yönlendirir ve durumun ciddiyetini değerlendirir. Kullanım Koşulları: - Kullanıcı bir tehlike (örneğin: "
            + "yangın"
            + ", "
            + "hırsızlık"
            + ", "
            + "bomba"
            + ", "
            + "kavga"
            + ", "
            + "yaralanma"
            + ") bildirdiğinde. - Kullanıcı doğrudan "
            + "güvenlik"
            + " veya "
            + "acil durum"
            + " gibi anahtar kelimelerle yardım istediğinde. - Kullanıcının davranışları veya sorguları endişe verici veya şüpheli olduğunda. Örnekler: "
            + "Acil yardım lazım, birisi bayıldı."
            + ", "
            + "Burası yanıyor, itfaiye gerekli!"
            + ", "
            + "Güvenlik çağırın!"
            + ", "
            + "Şu an odamda bir hırsız var."
            + " Bu araç bir tehdit algıladığında, hızla yanıt vererek durumu ciddiye almalı ve gerekli aksiyonları almalıdır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "details": {"type": "object"},
                },
            },
        },
    },
]


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
)
modelId = os.getenv("MODEL_ID", "qwen2.5:7b")

# modelId = "gpt-oss:20b" localde yavaş çalışıyor.
# llama3.1:8b yeterli gibi ancak bazen saçma cevaplar verebiliyor

# qwen2.5:7b iyi gibi gayet
# modelId = "qwen3:8b" güzel cevaplar ancak uzun düşünme süresi
# modelId = "RefinedNeuro/RN_TR_R2" fazla reasoning
def _unwrap_and_count(res) -> dict:
    try:
        data = res.get("data") if isinstance(res, dict) else None
        if isinstance(data, (list, tuple)):
            count = len(data)
        elif data:
            count = 1
        else:
            count = 0
        return {"found": count > 0, "count": count}
    except Exception:
        return {"found": False, "count": 0}


toolMap = {
    "findUserByName": lambda args: _unwrap_and_count(
        findUserByNameFn(args.get("employeeName"))
    ),
    "verifyUser": lambda args: _unwrap_and_count(
        verifyUserFn(args.get("employeeName"), args.get("password"))
    ),
    "findDeliveries": lambda args: _unwrap_and_count(
        findDeliveriesFn(
            args.get("company"), args.get("recipient"), args.get("status")
        )
    ),
    "findMeeting": lambda args: _unwrap_and_count(
        findMeetingFn(args.get("host"), args.get("guest"), args.get("time"))
    ),
    "signalDoor": lambda args: (
        signalDoorFn(args.get("action"), args.get("person"))
        or {"ok": True, "action": args.get("action")}
    ),
    "alertSecurity": lambda args: (
        alertSecurityFn(args.get("reason"), args.get("details"))
        or {"ok": True, "reason": args.get("reason")}
    ),
}


def runAgent(
    userInput: dict, decisionOnly: bool = True, history: list | None = None
) -> str:
    user_text = None
    try:
        if isinstance(userInput, dict):
            user_text = userInput.get("text")
    except Exception:
        user_text = None

    messages = [{"role": "system", "content": agentSystemPrompt}]

    if history:
        for turn in history:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": user_text or json.dumps(userInput, ensure_ascii=False),
        }
    )

    lastText = ""
    used_tool_this_turn = False
    safeguard_attempts = 0

    while True:
        resp = client.chat.completions.create(
            model=modelId,
            tools=tools,
            tool_choice="required",
            messages=messages,
            temperature=0,
        )

        choice = resp.choices[0]
        msg = choice.message

        assistantPayload = {"role": "assistant"}
        if getattr(msg, "content", None):
            assistantPayload["content"] = msg.content
            lastText = msg.content or lastText
        if getattr(msg, "tool_calls", None):
            assistantPayload["tool_calls"] = msg.tool_calls
        messages.append(assistantPayload)

        if getattr(msg, "tool_calls", None):
            used_tool_this_turn = True
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                try:
                    result = toolMap.get(name, lambda a: {"error": "tool_not_allowed"})(
                        args
                    )
                    content = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )
                except Exception as e:
                    content = json.dumps(
                        {"error": "tool_error", "message": str(e)}, ensure_ascii=False
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": content,
                    }
                )
            continue

        if getattr(msg, "content", None):
            # Enforce at least one tool call before finalizing
            if not used_tool_this_turn and safeguard_attempts < 2:
                safeguard_attempts += 1
                messages.append(
                    {
                        "role": "system",
                        "content": "Araç kullanımı zorunludur. Uygun backend fonksiyonunu çağırarak doğrulayın; emin değilseniz eksik bilgiyi netleştirin.",
                    }
                )
                continue
            return msg.content.strip()

        if choice.finish_reason in ("stop", "length", "content_filter"):
            return (lastText or "").strip()


if __name__ == "__main__":
    tests = [
        {"text": "Merhaba, Aras Kargo kuryesiyim. Umut Deniz'e teslimat var."},
        {"text": "Ben Mustafa Alkan, Burada çalışıyorum."},
        {"text": "Adım Arda Alper. Umut Deniz ile 4'te toplantım var"},
        {"text": "Kartım yok, kapıyı zorla açarım."},
        {"text": "Naber nasılsın?"},
        {"text": "Mercimek çorbası nasıl yapılır."},
        {"text": "Bugün Umut Deniz'in kargosu var mı?"},
    ]
    for t in tests:
        print(json.dumps(runAgent(t), ensure_ascii=False, indent=2))
        print("----------------------------------------------------")
