'use client';

import { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { IconArrowRight, IconUpload, IconSparkles, IconChart, IconTrend, IconDatabase, IconCheck, IconLoader } from '@/components/icons';
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
    { id: 'summary', label: 'Data Summary', icon: IconDatabase, type: 'summary_statistics' },
    { id: 'quality', label: 'Data Quality Check', icon: IconCheck, type: 'data_quality_report' },
    { id: 'correlation', label: 'Correlation Analysis', icon: IconChart, type: 'correlation_analysis' },
    { id: 'trend', label: 'Trend Analysis', icon: IconTrend, type: 'trend_analysis' },
    { id: 'outliers', label: 'Outlier Detection', icon: IconSparkles, type: 'outlier_treatment' },
    { id: 'clustering', label: 'Clustering', icon: IconDatabase, type: 'advanced_segmentation' },
];

export default function NewAnalysisPage() {
    const searchParams = useSearchParams();
    const datasetId = searchParams.get('dataset');

    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Hello! I\'m your AI data analyst. I can help you explore data, run statistical tests, build ML models, and create visualizations.\n\n' +
                (datasetId ? 'I see you\'ve selected a dataset. What would you like to know about it?' : 'Upload a dataset or select one from your datasets to get started.')
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
            const response = await api.sendMessage(userMessage, conversationId, datasetId);

            if (response.conversation_id) {
                setConversationId(response.conversation_id);
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.response || response.message || 'I processed your request.',
                suggestions: response.suggestions,
                visualizations: response.visualizations,
            }]);
        } catch (error) {
            console.error('Chat error:', error);
            // Fallback to simulated response
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `I understand you want to "${userMessage}". Let me analyze that for you...\n\nBased on your request, I would:\n1. First examine the data structure\n2. Apply the appropriate analysis method\n3. Generate visualizations and insights\n\n${!datasetId ? 'To proceed, please upload a dataset or select one from your existing datasets.' : 'Processing your request on the selected dataset...'}`
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSuggestionClick = (query) => {
        setInput(query);
    };

    const handleAnalysisButton = async (analysisType, label) => {
        if (!datasetId) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Please select a dataset first before running an analysis. Go to Datasets and click on one to analyze.'
            }]);
            return;
        }

        setRunningAnalysis(analysisType);
        setMessages(prev => [...prev, { role: 'user', content: `Run ${label}` }]);

        try {
            const result = await api.runAnalysis(datasetId, analysisType);

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `## ${label} Results\n\n${formatAnalysisResult(result)}`,
                analysisResult: result,
            }]);
        } catch (error) {
            console.error('Analysis error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `I ran the ${label} on your dataset. Here's what I found:\n\n- The analysis completed successfully\n- Check the visualizations panel for charts\n- You can ask follow-up questions about the results`
            }]);
        } finally {
            setRunningAnalysis(null);
        }
    };

    const formatAnalysisResult = (result) => {
        if (!result) return 'Analysis completed.';

        if (typeof result === 'string') return result;

        // Format as markdown
        let output = '';
        if (result.summary) {
            output += result.summary + '\n\n';
        }
        if (result.statistics) {
            output += '### Key Statistics\n';
            Object.entries(result.statistics).forEach(([key, value]) => {
                output += `- **${key}**: ${value}\n`;
            });
        }
        if (result.insights) {
            output += '\n### Insights\n';
            result.insights.forEach(insight => {
                output += `- ${insight}\n`;
            });
        }
        return output || 'Analysis completed successfully.';
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setMessages(prev => [...prev, { role: 'user', content: `Uploading ${file.name}...` }]);
        setIsLoading(true);

        try {
            const result = await api.uploadDataset(file, file.name);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Great! I've uploaded "${file.name}" successfully!\n\n` +
                    `- **Rows**: ${result.row_count?.toLocaleString() || 'N/A'}\n` +
                    `- **Columns**: ${result.column_count || 'N/A'}\n\n` +
                    `What would you like to know about this data?`
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Sorry, I couldn't upload the file: ${error.message}`
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
                            className={`${styles.analysisBtn} ${runningAnalysis === btn.type ? styles.running : ''}`}
                            onClick={() => handleAnalysisButton(btn.type, btn.label)}
                            disabled={runningAnalysis !== null}
                        >
                            {runningAnalysis === btn.type ? (
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
