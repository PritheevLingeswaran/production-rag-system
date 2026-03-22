const BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api/v1`
  : "/api/v1";

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string>),
  };

  if (API_KEY) {
    headers["x-api-key"] = API_KEY;
  }

  if (!(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${BASE}${path}`, { ...init, headers });

  if (!res.ok) {
    const message = await parseErrorMessage(res);
    throw new ApiError(res.status, message || `HTTP ${res.status}`);
  }

  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

async function parseErrorMessage(res: Response) {
  try {
    const payload = await res.json();
    if (typeof payload?.error?.message === "string") return payload.error.message;
    if (typeof payload?.detail?.message === "string") return payload.detail.message;
    if (typeof payload?.detail === "string") return payload.detail;
  } catch {
    // Fall back to plain text below.
  }

  return await res.text().catch(() => res.statusText);
}

export const api = {
  get: <T>(path: string) => request<T>(path, { method: "GET" }),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  upload: <T>(path: string, form: FormData) =>
    request<T>(path, { method: "POST", body: form }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
