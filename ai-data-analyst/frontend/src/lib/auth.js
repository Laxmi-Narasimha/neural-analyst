let _cachedTokens = null;

const ACCESS_TOKEN_KEY = 'na_access_token';
const REFRESH_TOKEN_KEY = 'na_refresh_token';

export function getTokens() {
  if (_cachedTokens) return _cachedTokens;
  if (typeof window === 'undefined') return null;

  const accessToken = window.localStorage.getItem(ACCESS_TOKEN_KEY);
  const refreshToken = window.localStorage.getItem(REFRESH_TOKEN_KEY);
  if (!accessToken) return null;

  _cachedTokens = { accessToken, refreshToken };
  return _cachedTokens;
}

export function getAccessToken() {
  return getTokens()?.accessToken || null;
}

export function setTokens(accessToken, refreshToken) {
  _cachedTokens = { accessToken, refreshToken };
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  if (refreshToken) window.localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function clearTokens() {
  _cachedTokens = null;
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  window.localStorage.removeItem(REFRESH_TOKEN_KEY);
}

