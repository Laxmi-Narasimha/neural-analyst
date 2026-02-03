// AI Enterprise Data Analyst - API Client
// Connects frontend to backend services

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class APIClient {
    constructor() {
        this.baseUrl = API_BASE_URL;
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

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
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

        const response = await fetch(`${this.baseUrl}/datasets/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(error.detail || 'Upload failed');
        }

        return await response.json();
    }

    async listDatasets(page = 1, pageSize = 20, search = '') {
        const params = new URLSearchParams({ page, page_size: pageSize });
        if (search) params.append('search', search);
        const response = await this.request(`/datasets?${params}`);
        return this._normalizeListResponse(response);
    }

    async getDataset(datasetId) {
        return this.request(`/datasets/${datasetId}`);
    }

    async deleteDataset(datasetId) {
        return this.request(`/datasets/${datasetId}`, { method: 'DELETE' });
    }

    async processDataset(datasetId) {
        return this.request(`/datasets/${datasetId}/process`, { method: 'POST' });
    }

    // ============================================================================
    // Chat / Analysis APIs
    // ============================================================================

    async sendMessage(message, conversationId = null, datasetId = null, context = {}) {
        return this.request('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                conversation_id: conversationId,
                dataset_id: datasetId,
                context,
            }),
        });
    }

    async listConversations(page = 1, pageSize = 20) {
        const response = await this.request(`/chat/conversations?page=${page}&page_size=${pageSize}`);
        return this._normalizeListResponse(response);
    }

    async getConversation(conversationId) {
        return this.request(`/chat/conversations/${conversationId}`);
    }

    async deleteConversation(conversationId) {
        return this.request(`/chat/conversations/${conversationId}`, { method: 'DELETE' });
    }

    // ============================================================================
    // Analytics APIs
    // ============================================================================

    async runAnalysis(datasetId, analysisType, config = {}) {
        return this.request('/analytics/run', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                analysis_type: analysisType,
                config,
            }),
        });
    }

    async getAnalysisResult(analysisId) {
        return this.request(`/analytics/${analysisId}`);
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

    async importFromConnection(connectionId, tableName, datasetName) {
        return this.request(`/connections/${connectionId}/import`, {
            method: 'POST',
            body: JSON.stringify({ table_name: tableName, dataset_name: datasetName }),
        });
    }

    // ============================================================================
    // ML APIs
    // ============================================================================

    async trainModel(datasetId, modelType, targetColumn, features = [], config = {}) {
        return this.request('/ml/train', {
            method: 'POST',
            body: JSON.stringify({
                dataset_id: datasetId,
                model_type: modelType,
                target_column: targetColumn,
                features,
                config,
            }),
        });
    }

    async predict(modelId, data) {
        return this.request(`/ml/predict/${modelId}`, {
            method: 'POST',
            body: JSON.stringify({ data }),
        });
    }

    async listModels() {
        const response = await this.request('/ml/models');
        return this._normalizeListResponse(response);
    }
}

// Singleton instance
const api = new APIClient();

export default api;
export { APIClient };
