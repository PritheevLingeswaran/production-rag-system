import { api } from "./client";
import type { Settings } from "../types";

export const settingsApi = {
  get: () => api.get<Settings>("/settings"),
};
