const sanitizeUrl = (url) => url.replace(/\/$/, "");

const getDefaultApiBaseUrl = () => {
    if (typeof window !== "undefined" && process.env.NODE_ENV === "production") {
        return `${window.location.origin.replace(/\/$/, "")}/api`;
    }
    return "http://localhost:5000/api";
};

const envBaseUrl = process.env.REACT_APP_API_BASE_URL?.trim();

export const API_BASE_URL = sanitizeUrl(envBaseUrl && envBaseUrl.length ? envBaseUrl : getDefaultApiBaseUrl());

const normalizePath = (path) => (path.startsWith("/") ? path : `/${path}`);

export const buildApiUrl = (path) => {
    return `${API_BASE_URL}${normalizePath(path)}`;
};

export const buildWsUrl = (path) => {
    const httpUrl = `${API_BASE_URL}${normalizePath(path)}`;
    if (httpUrl.startsWith("https://")) {
        return `wss://${httpUrl.slice("https://".length)}`;
    }
    if (httpUrl.startsWith("http://")) {
        return `ws://${httpUrl.slice("http://".length)}`;
    }
    return httpUrl;
};
