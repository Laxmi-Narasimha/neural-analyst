'use client';

import { useState } from 'react';
import { IconChat, IconAlert, IconCheck, IconChevronDown, IconChevronUp } from '@/components/icons';

interface Question {
    id: string;
    text: string;
    question?: string; // Alternative field name from backend
    priority?: 'critical' | 'high' | 'medium' | 'low';
    failure_mode?: string;
    expected_evidence?: string;
    context?: string;
}

interface QuestionsFormProps {
    questions: Question[];
    onSubmit: (answers: Record<string, string>) => void;
    isSubmitting: boolean;
}

export default function QuestionsForm({ questions, onSubmit, isSubmitting }: QuestionsFormProps) {
    const [answers, setAnswers] = useState<Record<string, string>>({});
    const [expandedEvidence, setExpandedEvidence] = useState<Set<string>>(new Set());
    const [validationErrors, setValidationErrors] = useState<string[]>([]);

    const getPriorityConfig = (priority?: string) => {
        switch (priority) {
            case 'critical':
                return {
                    icon: '🚨',
                    color: 'text-red-500',
                    bgColor: 'bg-red-500/10',
                    borderColor: 'border-red-500/30',
                    label: 'Critical'
                };
            case 'high':
                return {
                    icon: '⚠️',
                    color: 'text-orange-500',
                    bgColor: 'bg-orange-500/10',
                    borderColor: 'border-orange-500/30',
                    label: 'High'
                };
            case 'medium':
                return {
                    icon: '📋',
                    color: 'text-yellow-500',
                    bgColor: 'bg-yellow-500/10',
                    borderColor: 'border-yellow-500/30',
                    label: 'Medium'
                };
            default:
                return {
                    icon: '💡',
                    color: 'text-blue-400',
                    bgColor: 'bg-blue-500/10',
                    borderColor: 'border-blue-500/30',
                    label: 'Low'
                };
        }
    };

    const toggleEvidence = (id: string) => {
        setExpandedEvidence(prev => {
            const newSet = new Set(prev);
            if (newSet.has(id)) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        // Validate critical questions
        const criticalQuestions = questions.filter(q => q.priority === 'critical');
        const unanswered = criticalQuestions.filter(q => !answers[q.id]?.trim());

        if (unanswered.length > 0) {
            setValidationErrors(unanswered.map(q => (q.text || q.question || '').slice(0, 60) + '...'));
            return;
        }

        setValidationErrors([]);
        onSubmit(answers);
    };

    const handleChange = (id: string, value: string) => {
        setAnswers(prev => ({
            ...prev,
            [id]: value
        }));
        // Clear errors when user starts typing
        if (validationErrors.length > 0) {
            setValidationErrors([]);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <div className="flex items-start gap-3">
                    <IconChat className="text-blue-400 mt-1 flex-shrink-0" size={20} />
                    <div>
                        <h3 className="text-blue-400 font-semibold">Clarification Needed</h3>
                        <p className="text-sm text-gray-300 mt-1">
                            To ensure the highest quality assessment, please answer the following questions about your data and goals.
                            <span className="text-red-400 ml-2">Questions marked Critical are required.</span>
                        </p>
                    </div>
                </div>
            </div>

            {validationErrors.length > 0 && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                        <IconAlert className="text-red-500 mt-0.5 flex-shrink-0" size={20} />
                        <div>
                            <h4 className="text-red-400 font-semibold">Please answer all critical questions:</h4>
                            <ul className="mt-2 space-y-1">
                                {validationErrors.map((err, i) => (
                                    <li key={i} className="text-sm text-red-300">• {err}</li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
                {questions.map((q, index) => {
                    const priorityConfig = getPriorityConfig(q.priority);
                    const questionText = q.text || q.question || '';

                    return (
                        <div
                            key={q.id || index}
                            className={`border rounded-xl p-5 transition-all ${priorityConfig.borderColor} ${priorityConfig.bgColor}`}
                        >
                            {/* Question Header */}
                            <div className="flex items-start justify-between gap-4 mb-3">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="text-lg">{priorityConfig.icon}</span>
                                        <span className={`text-xs font-semibold uppercase ${priorityConfig.color}`}>
                                            {priorityConfig.label} Priority
                                        </span>
                                        {q.failure_mode && (
                                            <span className="text-xs text-gray-500 bg-gray-700/50 px-2 py-0.5 rounded">
                                                {q.failure_mode}
                                            </span>
                                        )}
                                    </div>
                                    <label className="block text-base font-medium text-gray-100">
                                        {index + 1}. {questionText}
                                    </label>
                                </div>
                            </div>

                            {/* Expected Evidence Expander */}
                            {q.expected_evidence && (
                                <div className="mb-3">
                                    <button
                                        type="button"
                                        onClick={() => toggleEvidence(q.id)}
                                        className="flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
                                    >
                                        {expandedEvidence.has(q.id) ? (
                                            <IconChevronUp size={16} />
                                        ) : (
                                            <IconChevronDown size={16} />
                                        )}
                                        What we're looking for
                                    </button>
                                    {expandedEvidence.has(q.id) && (
                                        <div className="mt-2 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                                            <p className="text-sm text-gray-300">{q.expected_evidence}</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Context */}
                            {q.context && (
                                <p className="text-xs text-gray-500 italic mb-3">{q.context}</p>
                            )}

                            {/* Answer Input */}
                            <textarea
                                value={answers[q.id] || ''}
                                onChange={(e) => handleChange(q.id, e.target.value)}
                                className="w-full bg-[#1a1a1a] border border-gray-600 rounded-lg p-3 text-gray-200 focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none transition-all placeholder-gray-600 min-h-[100px] resize-y"
                                placeholder="Please provide as much detail as possible..."
                                required={q.priority === 'critical'}
                            />
                        </div>
                    );
                })}

                <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold py-4 px-6 rounded-xl shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 text-lg"
                >
                    {isSubmitting ? (
                        <>
                            <span className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                            Processing Answers...
                        </>
                    ) : (
                        <>
                            <IconCheck size={20} />
                            Submit Answers & Continue Analysis
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
