'use client';

import { useState, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { IconUpload, IconSparkles, IconLoader, IconCheck, IconAlert, IconDownload, IconRefresh, IconChevronDown, IconChevronUp, IconClock, IconDatabase, IconBrain } from '@/components/icons';
import QuestionsForm from './QuestionsForm';
import ReadinessBadge from './ReadinessBadge';
import toast from 'react-hot-toast';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TechnicalDetails {
    namespace?: string;
    files_processed?: number;
    chunks_created?: number;
    llm_calls_used?: number;
    processing_time?: string;
}

interface Recommendation {
    issue: string;
    priority: 'critical' | 'high' | 'medium' | 'low';
    category?: string;
    details?: string;
    impact?: string;
    suggested_actions?: string[];
}

interface ValidationResults {
    success: boolean;
    session_id?: string;
    readiness_level?: string;
    composite_score?: number;
    executive_summary?: string;
    markdown_report?: string;
    top_recommendations?: Recommendation[];
    next_steps?: string[];
    technical_details?: TechnicalDetails;
    step?: string;
    questions?: any[];
    error?: string;
}

export default function QualityDashboard() {
    const [step, setStep] = useState<'initial' | 'processing' | 'questions' | 'results' | 'error'>('initial');
    const [datasetGoal, setDatasetGoal] = useState('');
    const [domain, setDomain] = useState('general');
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [questions, setQuestions] = useState<any[]>([]);
    const [results, setResults] = useState<ValidationResults | null>(null);
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']));

    const domains = [
        { value: 'general', label: 'General Purpose' },
        { value: 'automotive', label: 'Automotive Industry' },
        { value: 'manufacturing', label: 'Manufacturing' },
        { value: 'real_estate', label: 'Real Estate' }
    ];

    const toggleSection = (section: string) => {
        setExpandedSections(prev => {
            const newSet = new Set(prev);
            if (newSet.has(section)) {
                newSet.delete(section);
            } else {
                newSet.add(section);
            }
            return newSet;
        });
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        if (e.dataTransfer.files) {
            setFiles(Array.from(e.dataTransfer.files));
        }
    }, []);

    const startAnalysis = async () => {
        if (!datasetGoal || files.length === 0) {
            toast.error('Please provide a goal and upload files');
            return;
        }

        setStep('processing');
        setUploading(true);

        try {
            const filePaths = files.map(f => `uploads/${f.name}`);

            const response = await axios.post(`${API_BASE}/api/v1/quality/validate`, {
                goal: datasetGoal,
                domain: domain,
                files: filePaths
            });

            const data = response.data;
            setSessionId(data.session_id);

            if (data.step === 'clarifying_questions') {
                setQuestions(data.questions || []);
                setStep('questions');
            } else if (data.success) {
                setResults(data);
                setStep('results');
            } else {
                toast.error('Analysis failed: ' + (data.error || 'Unknown error'));
                setStep('error');
            }
        } catch (error: any) {
            console.error('Analysis error:', error);
            toast.error(error.response?.data?.detail || 'Failed to start analysis');
            setStep('error');
        } finally {
            setUploading(false);
        }
    };

    const submitAnswers = async (answers: Record<string, string>) => {
        setStep('processing');

        try {
            const response = await axios.post(`${API_BASE}/api/v1/quality/continue`, {
                session_id: sessionId,
                answers: answers
            });

            const data = response.data;

            if (data.success) {
                setResults(data);
                setStep('results');
            } else {
                toast.error('Validation failed: ' + (data.error || 'Unknown error'));
                setStep('error');
            }
        } catch (error: any) {
            console.error('Validation error:', error);
            toast.error(error.response?.data?.detail || 'Failed to submit answers');
            setStep('questions');
        }
    };

    const downloadReport = (format: 'markdown' | 'json') => {
        if (!results) return;

        let content: string;
        let filename: string;
        let mimeType: string;

        if (format === 'markdown') {
            content = results.markdown_report || '';
            filename = `quality-report-${sessionId?.slice(0, 8)}.md`;
            mimeType = 'text/markdown';
        } else {
            content = JSON.stringify(results, null, 2);
            filename = `quality-report-${sessionId?.slice(0, 8)}.json`;
            mimeType = 'application/json';
        }

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast.success(`Downloaded ${filename}`);
    };

    const resetAnalysis = () => {
        setStep('initial');
        setDatasetGoal('');
        setFiles([]);
        setSessionId(null);
        setQuestions([]);
        setResults(null);
    };

    const getPriorityIcon = (priority: string) => {
        switch (priority) {
            case 'critical': return '🚨';
            case 'high': return '⚠️';
            case 'medium': return '📋';
            default: return '💡';
        }
    };

    return (
        <div className="relative">
            <AnimatePresence mode="wait">
                {/* Initial State - Input Form */}
                {step === 'initial' && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        key="initial"
                        className="space-y-6"
                    >
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            {/* Left Column - Configuration */}
                            <div className="space-y-5">
                                <div>
                                    <label className="block text-sm font-medium text-gray-400 mb-2">Target Domain</label>
                                    <select
                                        value={domain}
                                        onChange={(e) => setDomain(e.target.value)}
                                        className="w-full bg-[#2a2a2a] border border-gray-700 rounded-xl p-4 text-gray-200 focus:ring-2 focus:ring-cyan-500 outline-none transition-all"
                                    >
                                        {domains.map(d => (
                                            <option key={d.value} value={d.value}>{d.label}</option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-gray-400 mb-2">AI Assistant Goal</label>
                                    <textarea
                                        value={datasetGoal}
                                        onChange={(e) => setDatasetGoal(e.target.value)}
                                        placeholder="Example: I want to create an AI assistant that helps customers find the right car based on their needs, budget, and preferences. It should provide accurate pricing, availability, and detailed specifications."
                                        className="w-full bg-[#2a2a2a] border border-gray-700 rounded-xl p-4 text-gray-200 focus:ring-2 focus:ring-cyan-500 outline-none min-h-[160px] transition-all resize-y"
                                    />
                                    <p className="text-xs text-gray-500 mt-1">Be as specific as possible about what your AI assistant should do.</p>
                                </div>
                            </div>

                            {/* Right Column - File Upload */}
                            <div>
                                <label className="block text-sm font-medium text-gray-400 mb-2">Knowledge Base Files</label>
                                <div
                                    className="border-2 border-dashed border-gray-700 rounded-xl p-8 flex flex-col items-center justify-center text-center hover:border-cyan-500/50 transition-all bg-[#2a2a2a]/50 min-h-[240px] cursor-pointer"
                                    onDrop={handleDrop}
                                    onDragOver={(e) => e.preventDefault()}
                                    onClick={() => document.getElementById('file-upload')?.click()}
                                >
                                    <IconUpload className="text-gray-500 mb-4" size={48} />
                                    <input
                                        type="file"
                                        multiple
                                        onChange={handleFileChange}
                                        className="hidden"
                                        id="file-upload"
                                        accept=".pdf,.docx,.txt,.csv,.xlsx"
                                    />
                                    <span className="text-cyan-500 font-semibold hover:text-cyan-400">Upload files</span>
                                    <span className="text-gray-500 text-sm mt-1">or drag and drop</span>
                                    <p className="text-xs text-gray-600 mt-3">PDF, DOCX, TXT, CSV, XLSX supported</p>

                                    {files.length > 0 && (
                                        <div className="mt-6 w-full max-h-[100px] overflow-y-auto border-t border-gray-700 pt-4">
                                            {files.map((f, i) => (
                                                <div key={i} className="text-sm text-gray-300 flex items-center gap-2 py-1">
                                                    <IconCheck size={14} className="text-green-500 flex-shrink-0" />
                                                    <span className="truncate">{f.name}</span>
                                                    <span className="text-gray-600 text-xs">({(f.size / 1024).toFixed(1)} KB)</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div className="flex justify-end pt-4">
                            <button
                                onClick={startAnalysis}
                                disabled={!datasetGoal || files.length === 0}
                                className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold py-4 px-10 rounded-xl shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed flex items-center gap-3 text-lg"
                            >
                                <IconSparkles size={22} />
                                Start Validation
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Processing State */}
                {step === 'processing' && (
                    <motion.div
                        key="processing"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex flex-col items-center justify-center h-[450px]"
                    >
                        <div className="relative">
                            <div className="absolute inset-0 bg-cyan-500/20 blur-2xl rounded-full animate-pulse" />
                            <IconLoader size={72} className="text-cyan-500 animate-spin relative z-10" />
                        </div>
                        <h3 className="text-2xl font-semibold text-white mt-10">Analyzing Data Adequacy...</h3>
                        <p className="text-gray-400 mt-3 text-center max-w-md">
                            Running comprehensive quality checks, generating test queries, and analyzing coverage gaps.
                        </p>
                        <div className="flex gap-2 mt-8">
                            <span className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <span className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <span className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                    </motion.div>
                )}

                {/* Questions State */}
                {step === 'questions' && (
                    <motion.div
                        key="questions"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                    >
                        <QuestionsForm
                            questions={questions}
                            onSubmit={submitAnswers}
                            isSubmitting={false}
                        />
                    </motion.div>
                )}

                {/* Results State */}
                {step === 'results' && results && (
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-8"
                    >
                        {/* Header Stats Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            <div className="bg-gradient-to-br from-[#2a2a2a] to-[#222] p-5 rounded-xl border border-gray-700">
                                <p className="text-sm text-gray-400 mb-2">Readiness Status</p>
                                <ReadinessBadge
                                    level={results.readiness_level || 'UNKNOWN'}
                                    score={results.composite_score}
                                    size="lg"
                                />
                            </div>
                            <div className="bg-gradient-to-br from-[#2a2a2a] to-[#222] p-5 rounded-xl border border-gray-700">
                                <p className="text-sm text-gray-400 mb-1">Composite Score</p>
                                <p className="text-4xl font-bold text-white">
                                    {((results.composite_score || 0) * 100).toFixed(1)}%
                                </p>
                            </div>
                            <div className="bg-gradient-to-br from-[#2a2a2a] to-[#222] p-5 rounded-xl border border-gray-700">
                                <p className="text-sm text-gray-400 mb-1">Files Processed</p>
                                <div className="flex items-center gap-2">
                                    <IconDatabase size={24} className="text-blue-400" />
                                    <p className="text-4xl font-bold text-white">
                                        {results.technical_details?.files_processed || files.length}
                                    </p>
                                </div>
                            </div>
                            <div className="bg-gradient-to-br from-[#2a2a2a] to-[#222] p-5 rounded-xl border border-gray-700">
                                <p className="text-sm text-gray-400 mb-1">Processing Time</p>
                                <div className="flex items-center gap-2">
                                    <IconClock size={24} className="text-green-400" />
                                    <p className="text-2xl font-bold text-white">
                                        {results.technical_details?.processing_time || 'N/A'}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Executive Summary */}
                        {results.executive_summary && (
                            <div className="bg-[#2a2a2a] rounded-xl border border-gray-700 overflow-hidden">
                                <button
                                    onClick={() => toggleSection('summary')}
                                    className="w-full p-4 flex items-center justify-between bg-gray-900/50 hover:bg-gray-900/70 transition-colors"
                                >
                                    <h3 className="font-semibold text-lg text-white flex items-center gap-2">
                                        📝 Executive Summary
                                    </h3>
                                    {expandedSections.has('summary') ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
                                </button>
                                {expandedSections.has('summary') && (
                                    <div className="p-6 prose prose-invert max-w-none">
                                        <ReactMarkdown>{results.executive_summary}</ReactMarkdown>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Top Recommendations */}
                        {results.top_recommendations && results.top_recommendations.length > 0 && (
                            <div className="bg-[#2a2a2a] rounded-xl border border-gray-700 overflow-hidden">
                                <button
                                    onClick={() => toggleSection('recommendations')}
                                    className="w-full p-4 flex items-center justify-between bg-gray-900/50 hover:bg-gray-900/70 transition-colors"
                                >
                                    <h3 className="font-semibold text-lg text-white flex items-center gap-2">
                                        ⚠️ Top Recommendations
                                    </h3>
                                    {expandedSections.has('recommendations') ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
                                </button>
                                {expandedSections.has('recommendations') && (
                                    <div className="p-4 space-y-3">
                                        {results.top_recommendations.slice(0, 5).map((rec, i) => (
                                            <div
                                                key={i}
                                                className="bg-gray-800/50 rounded-lg p-4 border border-gray-700"
                                            >
                                                <div className="flex items-start gap-3">
                                                    <span className="text-xl">{getPriorityIcon(rec.priority)}</span>
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 flex-wrap">
                                                            <h4 className="font-semibold text-white">{rec.issue}</h4>
                                                            <span className={`text-xs px-2 py-0.5 rounded uppercase ${rec.priority === 'critical' ? 'bg-red-500/20 text-red-400' :
                                                                    rec.priority === 'high' ? 'bg-orange-500/20 text-orange-400' :
                                                                        rec.priority === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                                                            'bg-blue-500/20 text-blue-400'
                                                                }`}>
                                                                {rec.priority}
                                                            </span>
                                                            {rec.category && (
                                                                <span className="text-xs bg-gray-700 px-2 py-0.5 rounded text-gray-400">
                                                                    {rec.category}
                                                                </span>
                                                            )}
                                                        </div>
                                                        {rec.details && (
                                                            <p className="text-sm text-gray-400 mt-2">{rec.details}</p>
                                                        )}
                                                        {rec.impact && (
                                                            <p className="text-sm text-gray-500 mt-1">
                                                                <strong>Impact:</strong> {rec.impact}
                                                            </p>
                                                        )}
                                                        {rec.suggested_actions && rec.suggested_actions.length > 0 && (
                                                            <div className="mt-3">
                                                                <p className="text-xs text-gray-500 uppercase mb-1">Suggested Actions:</p>
                                                                <ul className="space-y-1">
                                                                    {rec.suggested_actions.map((action, j) => (
                                                                        <li key={j} className="text-sm text-gray-300 flex items-start gap-2">
                                                                            <span className="text-cyan-500 mt-1">•</span>
                                                                            {action}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Next Steps */}
                        {results.next_steps && results.next_steps.length > 0 && (
                            <div className="bg-[#2a2a2a] rounded-xl border border-gray-700 overflow-hidden">
                                <div className="p-4 bg-gray-900/50">
                                    <h3 className="font-semibold text-lg text-white flex items-center gap-2">
                                        🎯 Next Steps
                                    </h3>
                                </div>
                                <div className="p-4">
                                    <ul className="space-y-2">
                                        {results.next_steps.map((step, i) => (
                                            <li key={i} className="flex items-start gap-3 text-gray-300">
                                                <span className="w-6 h-6 bg-cyan-500/20 text-cyan-400 rounded-full flex items-center justify-center text-sm font-semibold flex-shrink-0">
                                                    {i + 1}
                                                </span>
                                                {step}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        )}

                        {/* Technical Details */}
                        {results.technical_details && (
                            <div className="bg-[#2a2a2a] rounded-xl border border-gray-700 overflow-hidden">
                                <button
                                    onClick={() => toggleSection('technical')}
                                    className="w-full p-4 flex items-center justify-between bg-gray-900/50 hover:bg-gray-900/70 transition-colors"
                                >
                                    <h3 className="font-semibold text-lg text-white flex items-center gap-2">
                                        🔧 Technical Details
                                    </h3>
                                    {expandedSections.has('technical') ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
                                </button>
                                {expandedSections.has('technical') && (
                                    <div className="p-4">
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-gray-800/50 p-3 rounded-lg">
                                                <p className="text-xs text-gray-500 uppercase">Namespace</p>
                                                <p className="text-sm text-gray-200 font-mono mt-1">
                                                    {results.technical_details.namespace || 'N/A'}
                                                </p>
                                            </div>
                                            <div className="bg-gray-800/50 p-3 rounded-lg">
                                                <p className="text-xs text-gray-500 uppercase">Chunks Created</p>
                                                <p className="text-lg text-white font-semibold mt-1">
                                                    {results.technical_details.chunks_created || 0}
                                                </p>
                                            </div>
                                            <div className="bg-gray-800/50 p-3 rounded-lg">
                                                <p className="text-xs text-gray-500 uppercase">LLM Calls</p>
                                                <p className="text-lg text-white font-semibold mt-1">
                                                    {results.technical_details.llm_calls_used || 0}
                                                </p>
                                            </div>
                                            <div className="bg-gray-800/50 p-3 rounded-lg">
                                                <p className="text-xs text-gray-500 uppercase">Session ID</p>
                                                <p className="text-sm text-gray-200 font-mono mt-1 truncate">
                                                    {sessionId?.slice(0, 12) || 'N/A'}...
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Full Report */}
                        {results.markdown_report && (
                            <div className="bg-[#2a2a2a] rounded-xl border border-gray-700 overflow-hidden">
                                <button
                                    onClick={() => toggleSection('report')}
                                    className="w-full p-4 flex items-center justify-between bg-gray-900/50 hover:bg-gray-900/70 transition-colors"
                                >
                                    <h3 className="font-semibold text-lg text-white">📊 Full Assessment Report</h3>
                                    {expandedSections.has('report') ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
                                </button>
                                {expandedSections.has('report') && (
                                    <div className="p-8 prose prose-invert max-w-none">
                                        <ReactMarkdown
                                            components={{
                                                h1: ({ node, ...props }) => <h1 className="text-2xl font-bold text-white mb-6 pb-2 border-b border-gray-700" {...props} />,
                                                h2: ({ node, ...props }) => <h2 className="text-xl font-bold text-cyan-400 mt-8 mb-4" {...props} />,
                                                h3: ({ node, ...props }) => <h3 className="text-lg font-semibold text-white mt-6 mb-3" {...props} />,
                                                ul: ({ node, ...props }) => <ul className="list-disc pl-5 space-y-2 mb-4" {...props} />,
                                                li: ({ node, ...props }) => <li className="text-gray-300" {...props} />,
                                                p: ({ node, ...props }) => <p className="text-gray-300 mb-4 leading-relaxed" {...props} />,
                                                strong: ({ node, ...props }) => <strong className="font-bold text-white" {...props} />,
                                                table: ({ node, ...props }) => <div className="overflow-x-auto my-6"><table className="min-w-full divide-y divide-gray-700" {...props} /></div>,
                                                thead: ({ node, ...props }) => <thead className="bg-gray-800" {...props} />,
                                                th: ({ node, ...props }) => <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider" {...props} />,
                                                td: ({ node, ...props }) => <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300 border-b border-gray-700" {...props} />,
                                            }}
                                        >
                                            {results.markdown_report}
                                        </ReactMarkdown>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Download & Actions */}
                        <div className="flex flex-wrap items-center justify-between gap-4 pt-4">
                            <div className="flex gap-3">
                                <button
                                    onClick={() => downloadReport('markdown')}
                                    className="flex items-center gap-2 px-5 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                                >
                                    <IconDownload size={18} />
                                    Download Markdown
                                </button>
                                <button
                                    onClick={() => downloadReport('json')}
                                    className="flex items-center gap-2 px-5 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                                >
                                    <IconDownload size={18} />
                                    Download JSON
                                </button>
                            </div>
                            <button
                                onClick={resetAnalysis}
                                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold rounded-lg shadow-lg shadow-cyan-500/20 transition-all"
                            >
                                <IconRefresh size={18} />
                                Start New Analysis
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Error State */}
                {step === 'error' && (
                    <motion.div
                        key="error"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex flex-col items-center justify-center h-[400px]"
                    >
                        <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mb-6">
                            <IconAlert size={40} className="text-red-500" />
                        </div>
                        <h3 className="text-xl font-semibold text-white">Analysis Failed</h3>
                        <p className="text-gray-400 mt-2 text-center max-w-md">
                            Something went wrong during the analysis. Please try again.
                        </p>
                        <button
                            onClick={resetAnalysis}
                            className="mt-8 flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold rounded-lg transition-all"
                        >
                            <IconRefresh size={18} />
                            Try Again
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
