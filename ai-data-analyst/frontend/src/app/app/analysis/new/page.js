'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { IconArrowRight, IconUpload, IconSparkles, IconChart, IconDatabase, IconCheck, IconLoader } from '@/components/icons';
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
    { id: 'preview', label: 'Preview Rows', icon: IconDatabase, plan: [{ operator: 'preview_rows', params: { limit: 25 } }] },
    { id: 'missing', label: 'Missingness', icon: IconCheck, plan: [{ operator: 'missingness_scan', params: {} }] },
    { id: 'correlation', label: 'Correlation', icon: IconChart, plan: [{ operator: 'correlation_matrix', params: { max_columns: 25 } }] },
    { id: 'outliers', label: 'Outliers', icon: IconSparkles, plan: [{ operator: 'outlier_scan', params: { max_columns: 25 } }] },
];

export default function NewAnalysisPage() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const datasetIdParam = searchParams.get('dataset');
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
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            // Call the real backend API
            const response = await api.sendMessage(userMessage, conversationId, activeDatasetId);

            if (response.conversation_id) {
                setConversationId(response.conversation_id);
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.content || 'I processed your request.',
                suggestions: response.suggestions,
                agentActions: response.agent_actions,
            }]);
        } catch (error) {
            console.error('Chat error:', error);
            // Fallback to simulated response
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `I understand you want to "${userMessage}". Let me analyze that for you...\n\nBased on your request, I would:\n1. First examine the data structure\n2. Apply the appropriate analysis method\n3. Generate visualizations and insights\n\n${!activeDatasetId ? 'To proceed, please upload a dataset or select one from your existing datasets.' : 'Processing your request on the selected dataset...'}`
            }]);
        } finally {
            setIsLoading(false);
        }
    };

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

    const handleAnalysisButton = async (btn) => {
        if (!activeDatasetId) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Please select a dataset first before running an analysis. Go to Datasets and click on one to analyze.'
            }]);
            return;
        }

        try {
            const ds = await api.getDataset(activeDatasetId);
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
            const result = await api.runDataSpeaks(activeDatasetId, btn.plan);

            const steps = result?.steps || [];
            const lines = [];
            lines.push(`${btn.label} complete.`);
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
                                {message.content}
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
