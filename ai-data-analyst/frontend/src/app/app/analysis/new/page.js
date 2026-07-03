'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { IconArrowRight, IconUpload, IconSparkles, IconChart, IconDatabase, IconCheck, IconLoader, IconTime } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

const suggestedQueries = [
    'Show me a summary of this dataset',
    'What are the correlations between variables?',
    'Find outliers and anomalies in the data',
    'Train a prediction model for the target variable',
    'Run an A/B test analysis',
];

const analysisButtons = [
    { id: 'data_speaks', label: 'Data Speaks (EDA)', icon: IconSparkles, plan: null },
    { id: 'schema', label: 'Schema', icon: IconDatabase, plan: [{ operator: 'schema_snapshot', params: {} }] },
    { id: 'risk', label: 'Privacy & Risk', icon: IconCheck, plan: [{ operator: 'privacy_risk_scan', params: {} }] },
    { id: 'preview', label: 'Preview Rows', icon: IconDatabase, plan: [{ operator: 'preview_rows', params: { limit: 25 } }] },
    { id: 'missing', label: 'Missingness', icon: IconCheck, plan: [{ operator: 'missingness_scan', params: {} }] },
    { id: 'missing_patterns', label: 'Missingness Patterns', icon: IconCheck, plan: [{ operator: 'missingness_patterns', params: {} }] },
    { id: 'uniqueness', label: 'Uniqueness', icon: IconCheck, plan: [{ operator: 'uniqueness_scan', params: { max_columns: 200 } }] },
    { id: 'text', label: 'Text Summary', icon: IconDatabase, plan: [{ operator: 'text_summary', params: { max_columns: 25 } }] },
    { id: 'trend', label: 'Trend (Auto)', icon: IconChart, plan: [{ operator: 'resample_aggregate', params: { freq: 'M', max_points: 200 } }] },
    { id: 'time_anomalies', label: 'Time Anomalies', icon: IconTime, plan: [{ operator: 'time_anomaly_scan', params: { freq: 'M', max_points: 200 } }] },
    { id: 'segments', label: 'Segments (Auto)', icon: IconDatabase, plan: [{ operator: 'segment_summary', params: { limit: 50 } }] },
    { id: 'segment_deep_dive', label: 'Segment Deep Dive', icon: IconDatabase, plan: [{ operator: 'segment_deep_dive', params: { limit: 10 } }] },
    { id: 'correlation', label: 'Correlation', icon: IconChart, plan: [{ operator: 'correlation_matrix', params: { max_columns: 25 } }] },
    { id: 'associations', label: 'Associations', icon: IconChart, plan: [{ operator: 'association_scan', params: { max_categorical_columns: 20, max_numeric_columns: 20, max_pairs: 200 } }] },
    { id: 'outliers', label: 'Outliers', icon: IconSparkles, plan: [{ operator: 'outlier_scan', params: { max_columns: 25 } }] },
    { id: 'outlier_explain', label: 'Explain Outliers', icon: IconSparkles, plan: [{ operator: 'outlier_explain', params: {} }] },
];

export default function NewAnalysisPage() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const datasetIdParam = searchParams.get('dataset');
    const conversationParam = searchParams.get('conversation');
    const promptParam = searchParams.get('prompt');
    const autoSendParam = searchParams.get('autoSend');
    const actionParam = searchParams.get('action');
    const autoRunParam = searchParams.get('autoRun');
    const [activeDatasetId, setActiveDatasetId] = useState(datasetIdParam);

    useEffect(() => {
        setActiveDatasetId(datasetIdParam);
    }, [datasetIdParam]);

    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hello! I\'m your AI data analyst. I can help you explore data, run statistical tests, build ML models, and create visualizations.\n\n' +
                (datasetIdParam ? 'I see you\'ve selected a dataset. What would you like to know about it?' : 'Upload a dataset or select one from your datasets to get started.')
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const [runningAnalysis, setRunningAnalysis] = useState(null);
    const [conversations, setConversations] = useState([]);
    const [loadingConversations, setLoadingConversations] = useState(false);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const promptConsumedRef = useRef(false);
    const actionConsumedRef = useRef(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        const loadConversations = async () => {
            try {
                setLoadingConversations(true);
                const { items } = await api.listConversations(1, 12);
                setConversations(items || []);
            } catch (e) {
                console.error('Failed to load conversations', e);
            } finally {
                setLoadingConversations(false);
            }
        };
        loadConversations();
    }, [conversationId]);

    useEffect(() => {
        const convId = String(conversationParam || '').trim();
        if (!convId || convId === conversationId) return;

        const loadConversation = async () => {
            try {
                setIsLoading(true);
                const data = await api.getConversation(convId);
                setConversationId(convId);
                if (data?.active_dataset_id) {
                    setActiveDatasetId(String(data.active_dataset_id));
                }
                const restored = (data?.messages || []).map((m) => ({
                    role: m.role,
                    content: m.content,
                    agentActions: m.agent_actions,
                }));
                if (restored.length) {
                    setMessages(restored);
                }
            } catch (e) {
                console.error('Failed to restore conversation', e);
            } finally {
                setIsLoading(false);
            }
        };
        loadConversation();
    }, [conversationParam]);

    const appendAssistantMessage = (response) => {
        const clarificationRequired = Boolean(response?.metadata?.clarification_required);
        setMessages((prev) => [
            ...prev,
            {
                role: 'assistant',
                content: response.content || 'I could not compute an answer for that yet.',
                suggestions: response.suggestions,
                agentActions: response.agent_actions,
                clarification: clarificationRequired ? response.clarification : null,
            },
        ]);
    };

    const sendChatMessage = async (userMessage, { context = {}, displayMessage = null } = {}) => {
        const text = String(userMessage || '').trim();
        if (!text || isLoading) return;

        setInput('');
        setMessages((prev) => [...prev, { role: 'user', content: displayMessage || text }]);
        setIsLoading(true);

        try {
            const response = await api.sendMessage(text, conversationId, activeDatasetId, context);

            if (response.conversation_id) {
                setConversationId(response.conversation_id);
                router.replace(
                    `/app/analysis/new?dataset=${activeDatasetId || ''}&conversation=${response.conversation_id}`
                );
            }

            appendAssistantMessage(response);
        } catch (error) {
            console.error('Chat error:', error);
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: `Request failed: ${error?.message || 'Unknown error'}`,
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleClarificationSelect = async (clarification, option) => {
        if (!clarification || isLoading) return;

        const answer = String(option?.value || option?.label || '').trim();
        if (!answer) return;

        const label = String(option?.label || option?.value || answer);
        await sendChatMessage(answer, {
            displayMessage: label,
            context: {
                clarification: {
                    question_id: clarification.question_id,
                    answer,
                },
            },
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        await sendChatMessage(input);
    };

    useEffect(() => {
        const p = String(promptParam || '').trim();
        if (!p) return;
        if (promptConsumedRef.current) return;
        promptConsumedRef.current = true;

        setInput(p);
        const auto = String(autoSendParam || '').toLowerCase();
        if (auto === '1' || auto === 'true') {
            // Defer to allow state to settle before sending.
            setTimeout(() => {
                sendChatMessage(p);
            }, 0);
        }
    }, [promptParam, autoSendParam]);

    const handleSuggestionClick = (query) => {
        setInput(query);
    };

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const waitForDatasetReady = async (datasetId, { timeoutMs = 2 * 60 * 1000 } = {}) => {
        const deadline = Date.now() + timeoutMs;
        let attempt = 0;

        while (Date.now() < deadline) {
            const dataset = await api.getDataset(datasetId);
            const status = String(dataset?.status || '').toLowerCase();

            if (status === 'ready') return dataset;
            if (status === 'error') {
                const msg = dataset?.error_message || 'Dataset processing failed';
                throw new Error(msg);
            }

            attempt += 1;
            const delay = Math.min(2500, 400 + attempt * 200);
            await sleep(delay);
        }

        throw new Error('Timed out waiting for dataset processing to complete');
    };

    const waitForAnalysisDone = async (analysisId, { timeoutMs = 5 * 60 * 1000 } = {}) => {
        const deadline = Date.now() + timeoutMs;
        let attempt = 0;

        while (Date.now() < deadline) {
            const analysis = await api.getAnalysis(analysisId);
            const status = String(analysis?.status || '').toLowerCase();

            if (status === 'completed') return analysis;
            if (status === 'failed') {
                const msg = analysis?.error_message || 'Analysis failed';
                throw new Error(msg);
            }
            if (status === 'cancelled') {
                throw new Error('Analysis was cancelled');
            }

            attempt += 1;
            const delay = Math.min(2500, 400 + attempt * 200);
            await sleep(delay);
        }

        throw new Error('Timed out waiting for analysis to complete');
    };

    const handleAnalysisButton = async (btn) => {
        if (!activeDatasetId) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Please select a dataset first before running an analysis. Go to Datasets and click on one to analyze.'
            }]);
            return;
        }

        let ds = null;
        try {
            ds = await api.getDataset(activeDatasetId);
            if (String(ds?.status || '').toLowerCase() !== 'ready') {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: 'That dataset is still processing. Please wait a moment and try again.'
                }]);
                return;
            }
        } catch (e) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'I could not verify dataset status right now. Please try again.'
            }]);
            return;
        }

        setRunningAnalysis(btn.id);
        setMessages(prev => [...prev, { role: 'user', content: `Run ${btn.label}` }]);

        try {
            const analysisName = `${btn.label}: ${ds?.name || activeDatasetId}`;
            const config = { sample_rows: 200_000, ...(Array.isArray(btn.plan) ? { plan: btn.plan } : {}) };
            const created = await api.createAnalysis(analysisName, activeDatasetId, 'eda', config);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Started analysis.\n- analysis_id: ${created?.id}\n- status: ${created?.status}\n\nYou can open it here: /app/analysis/${created?.id}`
            }]);

            const final = await waitForAnalysisDone(created?.id);
            const steps = final?.results?.steps || [];

            const lines = [];
            lines.push(`${btn.label} complete.`);
            lines.push('');
            lines.push(`Analysis ID: ${final?.id}`);
            lines.push(`Status: ${final?.status}`);
            lines.push('');
            for (const s of steps) {
                const summary = s?.summary;
                const summaryText = summary && typeof summary === 'object' ? JSON.stringify(summary) : (summary || '');
                lines.push(`- ${s.operator}${summaryText ? `: ${summaryText}` : ''}`);
                for (const a of s.artifacts || []) {
                    lines.push(`  - ${a.artifact_type}: ${a.name} (artifact_id=${a.artifact_id})`);
                }
            }

            setMessages(prev => [...prev, { role: 'assistant', content: lines.join('\n') }]);
        } catch (error) {
            console.error('Analysis error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Failed to run ${btn.label}: ${error?.message || 'Unknown error'}`
            }]);
        } finally {
            setRunningAnalysis(null);
        }
    };

    useEffect(() => {
        const actionId = String(actionParam || '').trim();
        if (!actionId) return;
        if (actionConsumedRef.current) return;

        const auto = String(autoRunParam || '').toLowerCase();
        if (!(auto === '1' || auto === 'true')) return;

        const btn = analysisButtons.find((b) => b.id === actionId);
        if (!btn) return;

        actionConsumedRef.current = true;
        handleAnalysisButton(btn);
    }, [actionParam, autoRunParam, activeDatasetId]);

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        e.target.value = '';

        setMessages(prev => [...prev, { role: 'user', content: `Uploading ${file.name}...` }]);
        setIsLoading(true);

        try {
            const uploaded = await api.uploadDataset(file, file.name);
            const datasetId = uploaded?.dataset_id;

            if (!datasetId) {
                throw new Error('Upload did not return a dataset_id');
            }

            setMessages(prev => [...prev, { role: 'assistant', content: 'Upload complete. Processing started...' }]);
            const processed = await waitForDatasetReady(datasetId);

            setActiveDatasetId(datasetId);
            setConversationId(null);
            router.replace(`/app/analysis/new?dataset=${datasetId}`);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Dataset ready.\n- Rows: ${(processed?.row_count || 0).toLocaleString()}\n- Columns: ${processed?.column_count || 0}\n\nAsk a question like: \"show missing values\".`
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Sorry, I couldn't upload/process the file: ${error.message}`
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className={styles.page}>
            <div className={styles.chatContainer}>
                {conversations.length > 0 && (
                    <div className={styles.conversationBar}>
                        <span className={styles.conversationLabel}>
                            {loadingConversations ? 'Loading chats...' : 'Recent conversations'}
                        </span>
                        <div className={styles.conversationList}>
                            {conversations.map((c) => (
                                <button
                                    key={c.id}
                                    type="button"
                                    className={`${styles.conversationChip} ${String(conversationId) === String(c.id) ? styles.conversationChipActive : ''}`}
                                    onClick={() => router.push(`/app/analysis/new?dataset=${c.active_dataset_id || activeDatasetId || ''}&conversation=${c.id}`)}
                                >
                                    {c.title || 'Conversation'}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                {/* Analysis Buttons */}
                <div className={styles.analysisButtons}>
                    {analysisButtons.map((btn) => (
                        <button
                            key={btn.id}
                            className={`${styles.analysisBtn} ${runningAnalysis === btn.id ? styles.running : ''}`}
                            onClick={() => handleAnalysisButton(btn)}
                            disabled={runningAnalysis !== null}
                        >
                            {runningAnalysis === btn.id ? (
                                <IconLoader size={16} className={styles.spinning} />
                            ) : (
                                <btn.icon size={16} />
                            )}
                            {btn.label}
                        </button>
                    ))}
                </div>

                {/* Messages */}
                <div className={styles.messages}>
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`${styles.message} ${styles[message.role]}`}
                        >
                            {message.role === 'assistant' && (
                                <div className={styles.avatar}>
                                    <IconSparkles size={16} />
                                </div>
                            )}
                            <div className={styles.messageContent}>
                                <div className={styles.messageText}>{message.content}</div>
                                {Array.isArray(message.clarification?.options) && message.clarification.options.length > 0 && (
                                    <div className={styles.clarificationOptions}>
                                        {message.clarification.options.map((option, optionIndex) => {
                                            const label = option?.label || option?.value || `Option ${optionIndex + 1}`;
                                            return (
                                                <button
                                                    key={`${message.clarification.question_id || 'clarify'}-${optionIndex}`}
                                                    type="button"
                                                    className={styles.clarificationBtn}
                                                    onClick={() => handleClarificationSelect(message.clarification, option)}
                                                    disabled={isLoading}
                                                >
                                                    <span className={styles.clarificationIndex}>{optionIndex + 1}</span>
                                                    {label}
                                                </button>
                                            );
                                        })}
                                    </div>
                                )}
                                {Array.isArray(message.suggestions) && message.suggestions.length > 0 && (
                                    <div className={styles.messageSuggestions}>
                                        {message.suggestions.map((suggestion, suggestionIndex) => (
                                            <button
                                                key={`${index}-suggestion-${suggestionIndex}`}
                                                type="button"
                                                className={styles.suggestionBtn}
                                                onClick={() => handleSuggestionClick(suggestion)}
                                                disabled={isLoading}
                                            >
                                                {suggestion}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className={`${styles.message} ${styles.assistant}`}>
                            <div className={styles.avatar}>
                                <IconSparkles size={16} />
                            </div>
                            <div className={styles.messageContent}>
                                <span className={styles.typing}>
                                    <span></span><span></span><span></span>
                                </span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Suggestions */}
                {messages.length === 1 && (
                    <div className={styles.suggestions}>
                        {suggestedQueries.map((query, i) => (
                            <button
                                key={i}
                                className={styles.suggestionBtn}
                                onClick={() => handleSuggestionClick(query)}
                            >
                                {query}
                            </button>
                        ))}
                    </div>
                )}

                {/* Input */}
                <form onSubmit={handleSubmit} className={styles.inputForm}>
                    <div className={styles.inputWrapper}>
                        <button
                            type="button"
                            className={styles.attachBtn}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <IconUpload size={18} />
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".csv,.xlsx,.xls,.json,.parquet"
                            onChange={handleFileUpload}
                            style={{ display: 'none' }}
                        />
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask anything about your data..."
                            className={styles.input}
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            className={styles.sendBtn}
                            disabled={!input.trim() || isLoading}
                        >
                            <IconArrowRight size={18} />
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
