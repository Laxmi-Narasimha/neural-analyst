// AI Enterprise Data Analyst - API Client
// Connects frontend to backend services

import { clearTokens, getAccessToken, setTokens } from './auth';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// The backend stores and sends UTC timestamps without a zone ("2026-11-25T06:37:27.6"). Browsers read
// such strings as local time, so "just now" showed as "5 hours ago" east of UTC. Mark them as UTC once,
// here, for every response.
const NAIVE_ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$/;
export function markUtc(value) {
    if (typeof value === 'string') return NAIVE_ISO.test(value) ? `${value}Z` : value;
    if (Array.isArray(value)) return value.map(markUtc);
    if (value && typeof value === 'object') {
        const out = {};
        for (const [k, v] of Object.entries(value)) out[k] = markUtc(v);
        return out;
    }
    return value;
}

class APIClient {
    constructor() {
        this.baseUrl = API_BASE_URL;
    }

    _unwrapApiResponse(response) {
        if (!response || typeof response !== 'object') return response;
        if (Object.prototype.hasOwnProperty.call(response, 'data')) return response.data;
        return response;
    }

    _normalizeListResponse(response) {
        if (!response || typeof response !== 'object') {
            return { items: [], raw: response };
        }

        if (Array.isArray(response.items)) {
            return { items: response.items, pagination: response.pagination, raw: response };
        }

        if (Array.isArray(response.data)) {
            return { items: response.data, pagination: response.pagination, raw: response };
        }

        // Some endpoints return APIResponse envelopes: { status, data: {...} }
        if (response.data && Array.isArray(response.data.items)) {
            return { items: response.data.items, pagination: response.data.pagination, raw: response };
        }

        return { items: [], raw: response };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        const token = getAccessToken();
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { Authorization: `Bearer ${token}` } : {}),
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                if (response.status === 401) {
                    clearTokens();
                }
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                const detail = typeof error.detail === 'string' ? error.detail : (error.detail?.message || error.message);
                const failure = new Error(detail || `HTTP ${response.status}`);
                failure.status = response.status;
                failure.body = error;
                throw failure;
            }

            return markUtc(await response.json());
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    // ============================================================================
    // Dataset APIs
    // ============================================================================

    async uploadDataset(file, name, description = '', tags = []) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', name || file.name);
        if (description) formData.append('description', description);
        if (tags.length) formData.append('tags', tags.join(','));

        const token = getAccessToken();
        const response = await fetch(`${this.baseUrl}/datasets/upload`, {
            method: 'POST',
            body: formData,
            headers: token ? { Authorization: `Bearer ${token}` } : {},
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || 'Upload failed');
        }

        const json = markUtc(await response.json());
        return this._unwrapApiResponse(json);
    }

    async listDatasets(page = 1, pageSize = 20, search = '') {
        // The API serves at most 100 per page and rejects larger requests (422).
        const params = new URLSearchParams({ page, page_size: Math.min(Math.max(1, pageSize), 100) });
        if (search) params.append('search', search);
        const response = await this.request(`/datasets?${params}`);
        return this._normalizeListResponse(response);
    }

    async getDataset(datasetId) {
        const response = await this.request(`/datasets/${datasetId}`);
        return this._unwrapApiResponse(response);
    }

    async deleteDataset(datasetId) {
        const response = await this.request(`/datasets/${datasetId}`, { method: 'DELETE' });
        return this._unwrapApiResponse(response);
    }

    async processDataset(datasetId) {
        const response = await this.request(`/datasets/${datasetId}/process`, { method: 'POST' });
        return this._unwrapApiResponse(response);
    }

    async listDatasetVersions(datasetId, page = 1, pageSize = 50) {
        const params = new URLSearchParams({ page, page_size: pageSize });
        const response = await this.request(`/datasets/${datasetId}/versions?${params}`);
        return this._normalizeListResponse(response);
    }

    async suggestDatasetTransform(datasetId, { maxSteps = 8, includeDropColumns = true, includeStringNormalization = true } = {}) {
        const response = await this.request(`/datasets/${datasetId}/transform/suggest`, {
            method: 'POST',
            body: JSON.stringify({
                max_steps: Number(maxSteps),
                include_drop_columns: Boolean(includeDropColumns),
                include_string_normalization: Boolean(includeStringNormalization),
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async previewDatasetTransform(datasetId, steps, sampleRows = 50_000, previewRows = 25) {
        const response = await this.request(`/datasets/${datasetId}/transform/preview`, {
            method: 'POST',
            body: JSON.stringify({
                steps: Array.isArray(steps) ? steps : [],
                sample_rows: sampleRows,
                preview_rows: previewRows,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async applyDatasetTransform(datasetId, steps, { label = null, setAsCurrent = true } = {}) {
        const response = await this.request(`/datasets/${datasetId}/transform/apply`, {
            method: 'POST',
            body: JSON.stringify({
                steps: Array.isArray(steps) ? steps : [],
                label,
                set_as_current: Boolean(setAsCurrent),
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async activateDatasetVersion(datasetId, versionId) {
        const response = await this.request(`/datasets/${datasetId}/versions/${versionId}/activate`, { method: 'POST' });
        return this._unwrapApiResponse(response);
    }

    async queryDatasetSql(datasetId, query, { maxRows = null, timeoutSeconds = null } = {}) {
        const q = String(query || '').trim();
        const body = { query: q };
        if (maxRows != null && Number.isFinite(Number(maxRows))) body.max_rows = Number(maxRows);
        if (timeoutSeconds != null && Number.isFinite(Number(timeoutSeconds))) body.timeout_seconds = Number(timeoutSeconds);
        const response = await this.request(`/datasets/${datasetId}/query`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Chat / Analysis APIs
    // ============================================================================

    async sendMessage(message, conversationId = null, datasetId = null, context = {}) {
        const response = await this.request('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                conversation_id: conversationId,
                dataset_id: datasetId,
                context,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async listConversations(page = 1, pageSize = 20) {
        const response = await this.request(`/chat/conversations?page=${page}&page_size=${pageSize}`);
        return this._normalizeListResponse(response);
    }

    async getConversation(conversationId) {
        const response = await this.request(`/chat/conversations/${conversationId}`);
        return this._unwrapApiResponse(response);
    }

    async deleteConversation(conversationId) {
        return this.request(`/chat/conversations/${conversationId}`, { method: 'DELETE' });
    }

    // ============================================================================
    // Analytics APIs
    // ============================================================================

    async runAnalysis(datasetId, analysisType = 'eda', config = {}) {
        return this.createAnalysis(`${analysisType} analysis`, datasetId, analysisType, config);
    }

    async getAnalysisResult(analysisId) {
        return this.getAnalysis(analysisId);
    }

    // ============================================================================
    // Billing / Subscription APIs
    // ============================================================================

    async getSystemInfo() {
        const response = await this.request('/system/info');
        return this._unwrapApiResponse(response);
    }

    async getSubscriptionStatus() {
        const response = await this.request('/billing/status');
        return this._unwrapApiResponse(response);
    }

    async createCheckoutSession(plan, options = {}) {
        const response = await this.request('/billing/checkout', {
            method: 'POST',
            body: JSON.stringify({
                plan,
                success_url: options.successUrl,
                cancel_url: options.cancelUrl,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async openBillingPortal(returnUrl = null) {
        const response = await this.request('/billing/portal', {
            method: 'POST',
            body: JSON.stringify({ return_url: returnUrl }),
        });
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Database Connection APIs
    // ============================================================================

    async createConnection(config) {
        return this.request('/connections', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    }

    async listConnections() {
        const response = await this.request('/connections');
        return this._normalizeListResponse(response);
    }

    async testConnection(connectionId) {
        return this.request(`/connections/${connectionId}/test`, { method: 'POST' });
    }

    async deleteConnection(connectionId) {
        return this.request(`/connections/${connectionId}`, { method: 'DELETE' });
    }

    async queryDatabase(connectionId, query) {
        return this.request(`/connections/${connectionId}/query`, {
            method: 'POST',
            body: JSON.stringify({ query }),
        });
    }

    async listConnectionTables(connectionId) {
        return this.request(`/connections/${connectionId}/tables`);
    }

    async importFromConnection(connectionId, tableName, datasetName, options = {}) {
        return this.request(`/connections/${connectionId}/import`, {
            method: 'POST',
            body: JSON.stringify({ table_name: tableName, dataset_name: datasetName, ...options }),
        });
    }

    // ============================================================================
    // ML APIs
    // ============================================================================

    async trainModel(datasetId, targetColumn, options = {}) {
        const response = await this.request('/ml/train', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                target_column: targetColumn,
                task: options.task || null,
                optimize: options.optimize !== false,
                mode: options.mode || 'baseline',
                sample_rows: options.sampleRows || null,
                max_models: options.maxModels || null,
            }),
        });
        return this._unwrapApiResponse(response) ?? response;
    }

    async runAutoML(datasetId, targetColumn, options = {}) {
        const response = await this.request('/ml/automl', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                target_column: targetColumn,
                task: options.task || null,
                time_budget_minutes: options.timeBudgetMinutes || 60,
                max_models: options.maxModels || 10,
                sample_rows: options.sampleRows || null,
                optimize: options.optimize !== false,
            }),
        });
        return this._unwrapApiResponse(response) ?? response;
    }

    async inferMLTask(datasetId, options = {}) {
        const response = await this.request('/ml/task-inference', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                preferred_target: options.preferredTarget || null,
                sample_rows: options.sampleRows || 100000,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async getMLJob(jobId) {
        return this.request(`/ml/jobs/${jobId}`);
    }

    async predict(modelName, data, modelVersion = null) {
        const response = await this.request('/ml/predict', {
            method: 'POST',
            body: JSON.stringify({
                model_name: modelName,
                model_version: modelVersion,
                data,
            }),
        });
        return this._unwrapApiResponse(response) ?? response;
    }

    async listModels() {
        const response = await this.request('/ml/models');
        return this._normalizeListResponse(response);
    }

    // ============================================================================
    // Data Speaks / Artifacts
    // ============================================================================

    async runDataSpeaks(datasetId, plan = null, sampleRows = null) {
        const response = await this.request('/data-speaks/run', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                plan,
                sample_rows: sampleRows,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async getArtifactManifest(artifactId) {
        const response = await this.request(`/data-speaks/artifacts/${artifactId}`);
        return this._unwrapApiResponse(response);
    }

    async downloadArtifactData(artifactId) {
        const token = getAccessToken();
        const response = await fetch(`${this.baseUrl}/data-speaks/artifacts/${artifactId}/download`, {
            method: 'GET',
            headers: token ? { Authorization: `Bearer ${token}` } : {},
        });

        if (!response.ok) {
            if (response.status === 401) clearTokens();
            const error = await response.json().catch(() => ({ detail: 'Download failed' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.blob();
    }

    // ============================================================================
    // Analyses (persisted jobs)
    // ============================================================================

    async createAnalysis(name, datasetId, analysisType = 'eda', config = {}) {
        const response = await this.request('/analyses', {
            method: 'POST',
            body: JSON.stringify({
                name,
                dataset_id: datasetId,
                analysis_type: analysisType,
                config,
            }),
        });
        return this._unwrapApiResponse(response);
    }

    async listAnalyses(page = 1, pageSize = 20, datasetId = null) {
        const params = new URLSearchParams({ page, page_size: pageSize });
        if (datasetId) params.append('dataset_id', datasetId);
        const response = await this.request(`/analyses?${params}`);
        return this._normalizeListResponse(response);
    }

    async getAnalysis(analysisId) {
        const response = await this.request(`/analyses/${analysisId}`);
        return this._unwrapApiResponse(response);
    }

    streamAnalysisEvents(
        analysisId,
        { onMeta = null, onStep = null, onDone = null, onError = null, signal: externalSignal = null } = {}
    ) {
        const controller = externalSignal ? null : new AbortController();
        const signal = externalSignal || controller.signal;

        const token = getAccessToken();
        const url = `${this.baseUrl}/analyses/${analysisId}/events`;

        const run = async () => {
            let sawDone = false;
            const res = await fetch(url, {
                method: 'GET',
                headers: {
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                    Accept: 'text/event-stream',
                },
                signal,
            });

            if (!res.ok) {
                if (res.status === 401) clearTokens();
                const text = await res.text().catch(() => '');
                throw new Error(text || `HTTP ${res.status}`);
            }

            if (!res.body || typeof res.body.getReader !== 'function') {
                throw new Error('Streaming not supported by this browser');
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            const flush = () => {
                const normalized = buffer.replace(/\r\n/g, '\n');
                const parts = normalized.split('\n\n');
                buffer = parts.pop() || '';
                for (const part of parts) {
                    const lines = part.split('\n');
                    let eventName = 'message';
                    const dataLines = [];
                    for (const line of lines) {
                        if (!line) continue;
                        if (line.startsWith(':')) continue;
                        if (line.startsWith('event:')) {
                            eventName = line.slice('event:'.length).trim() || 'message';
                            continue;
                        }
                        if (line.startsWith('data:')) {
                            dataLines.push(line.slice('data:'.length).trimStart());
                        }
                    }

                    if (dataLines.length === 0) continue;
                    const dataRaw = dataLines.join('\n');
                    let data = null;
                    try {
                        data = JSON.parse(dataRaw);
                    } catch {
                        data = dataRaw;
                    }

                    if (eventName === 'analysis_meta') onMeta && onMeta(data);
                    else if (eventName === 'analysis_step') onStep && onStep(data);
                    else if (eventName === 'analysis_done') {
                        sawDone = true;
                        onDone && onDone(data);
                    }
                }
            };

            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    flush();
                }

                buffer += decoder.decode();
                flush();

                if (!signal?.aborted && !sawDone) {
                    throw new Error('Analysis event stream closed before completion');
                }
            } finally {
                try {
                    reader.releaseLock();
                } catch {
                    // ignore
                }
            }
        };

        run().catch((err) => {
            if (signal?.aborted) return;
            onError && onError(err);
        });

        return () => {
            try {
                controller && controller.abort();
            } catch {
                // ignore
            }
        };
    }

    async cancelAnalysis(analysisId) {
        const response = await this.request(`/analyses/${analysisId}/cancel`, { method: 'POST' });
        return this._unwrapApiResponse(response);
    }

    async exportAnalysisReport(analysisId, format = 'markdown') {
        const params = new URLSearchParams({ format });
        const response = await this.request(`/analyses/${analysisId}/export?${params}`, { method: 'POST' });
        return this._unwrapApiResponse(response);
    }

    async listAnalysisActions(analysisId) {
        const response = await this.request(`/analyses/${analysisId}/actions`);
        return this._unwrapApiResponse(response);
    }

    async runAnalysisAction(analysisId, actionId, params = {}) {
        const response = await this.request(`/analyses/${analysisId}/actions/run`, {
            method: 'POST',
            body: JSON.stringify({
                action_id: actionId,
                params: params && typeof params === 'object' ? params : {},
            }),
        });
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Artifacts / Reports
    // ============================================================================

    async listArtifacts(page = 1, pageSize = 20, filters = {}) {
        const params = new URLSearchParams({ page, page_size: pageSize });
        if (filters.type) params.append('type', filters.type);
        if (filters.dataset_id) params.append('dataset_id', filters.dataset_id);
        if (filters.operator_name) params.append('operator_name', filters.operator_name);
        const response = await this.request(`/artifacts?${params}`);
        return this._normalizeListResponse(response);
    }

    async getArtifact(artifactId) {
        const response = await this.request(`/artifacts/${artifactId}`);
        return this._unwrapApiResponse(response);
    }

    async getArtifactRows(artifactId, offset = 0, limit = 50) {
        const params = new URLSearchParams({
            offset: String(offset),
            limit: String(limit),
        });
        const response = await this.request(`/artifacts/${artifactId}/rows?${params}`);
        return this._unwrapApiResponse(response);
    }

    async createReportShare(artifactId, expiresDays = null) {
        const body = {};
        if (expiresDays != null) body.expires_days = expiresDays;
        const response = await this.request(`/shares/reports/${artifactId}`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
        return this._unwrapApiResponse(response);
    }

    async getSharedReport(shareToken) {
        const token = encodeURIComponent(String(shareToken || ''));
        const response = await this.request(`/public/reports/${token}`);
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Dashboard
    // ============================================================================

    async getDashboardSummary() {
        const response = await this.request('/dashboard/summary');
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Data Quality / Adequacy
    // ============================================================================

    async startQualityValidation(goal, domain = 'general', datasetId = null) {
        const response = await this.request('/quality/validate', {
            method: 'POST',
            body: JSON.stringify({
                goal,
                domain,
                dataset_id: datasetId,
            }),
        });
        return response;
    }

    async continueQualityValidation(sessionId, answers) {
        const response = await this.request('/quality/continue', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                answers,
            }),
        });
        return response;
    }

    async getQualityStatus(sessionId) {
        return this.request(`/quality/status/${sessionId}`);
    }

    async listQualityDomains() {
        return this.request('/quality/domains');
    }

    async getLatestQualitySession(datasetId) {
        const params = new URLSearchParams({ dataset_id: datasetId });
        const response = await this.request(`/quality/sessions/latest?${params}`);
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Jobs
    // ============================================================================

    async listJobs(page = 1, pageSize = 50, filters = {}) {
        const params = new URLSearchParams({ page, page_size: pageSize });
        if (filters.status) params.append('status', filters.status);
        if (filters.job_type) params.append('job_type', filters.job_type);
        if (filters.dataset_id) params.append('dataset_id', filters.dataset_id);
        const response = await this.request(`/jobs?${params}`);
        return this._normalizeListResponse(response);
    }

    async getJob(jobId) {
        const response = await this.request(`/jobs/${jobId}`);
        return this._unwrapApiResponse(response);
    }

    async cancelJob(jobId) {
        const response = await this.request(`/jobs/${jobId}/cancel`, { method: 'POST' });
        return this._unwrapApiResponse(response);
    }

    // ============================================================================
    // Auth APIs
    // ============================================================================

    async register(email, password, fullName, role = 'analyst', inviteCode = '') {
        return this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({
                email,
                password,
                full_name: fullName,
                role,
                invite_code: inviteCode,
            }),
        });
    }

    async login(email, password) {
        const tokens = await this.request('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password }),
        });
        if (tokens?.access_token) {
            setTokens(tokens.access_token, tokens.refresh_token);
        }
        return tokens;
    }

    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } finally {
            clearTokens();
        }
    }

    async getCurrentUser() {
        const response = await this.request('/auth/me');
        return this._unwrapApiResponse(response);
    }
}

// Singleton instance
const api = new APIClient();

export default api;
export { APIClient };
