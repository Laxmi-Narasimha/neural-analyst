'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { IconKey, IconShield, IconCheck, IconArrowRight, IconDatabase, IconSparkles } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

const providerGuides = [
    {
        id: 'openai',
        name: 'OpenAI',
        model: 'gpt-4o-mini',
        envVar: 'OPENAI_API_KEY',
        docs: 'https://platform.openai.com/api-keys',
    },
    {
        id: 'anthropic',
        name: 'Anthropic',
        model: 'claude-sonnet-4-20250514',
        envVar: 'ANTHROPIC_API_KEY',
        docs: 'https://console.anthropic.com/settings/keys',
    },
    {
        id: 'google',
        name: 'Google Gemini',
        model: 'gemini/gemini-1.5-flash',
        envVar: 'GEMINI_API_KEY',
        docs: 'https://aistudio.google.com/app/apikey',
    },
    {
        id: 'ollama',
        name: 'Ollama (local, free)',
        model: 'ollama/llama3',
        envVar: null,
        docs: 'https://ollama.com',
    },
];

function StatusPill({ ok, label }) {
    return (
        <span className={`${styles.statusPill} ${ok ? styles.statusOk : styles.statusWarn}`}>
            {ok ? <IconCheck size={14} /> : null}
            {label}
        </span>
    );
}

export default function SetupKeysPage() {
    const [info, setInfo] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedProvider, setSelectedProvider] = useState('openai');

    useEffect(() => {
        const load = async () => {
            try {
                setLoading(true);
                const data = await api.getSystemInfo();
                setInfo(data);
            } catch (e) {
                setError(e?.message || 'Could not reach the API. Start the backend first.');
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    const keys = info?.api_keys_present || {};
    const provider = providerGuides.find((p) => p.id === selectedProvider) || providerGuides[0];
    const llmReady = Boolean(info?.llm_configured);
    const compute = info?.compute || {};

    const envSnippet = [
        'DEPLOYMENT_MODE=self_host',
        'AUTH_MODE=local',
        `LLM_MODEL=${provider.model}`,
        provider.envVar ? `${provider.envVar}=your-key-here` : '# Ollama: run `ollama serve` — no key needed',
        'JOB_EXECUTOR=local',
    ].join('\n');

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className={styles.content}
                >
                    <div className={styles.header}>
                        <div className={styles.iconWrapper}>
                            <IconKey size={32} />
                        </div>
                        <h1 className={styles.title}>
                            <span className={styles.gradient}>Self-host</span> & BYOK
                        </h1>
                        <p className={styles.subtitle}>
                            Run Neural Analyst on your machine or cloud. You bring API keys and compute —
                            we never host your data or training jobs.
                        </p>
                    </div>

                    {loading && <p className={styles.hint}>Checking deployment status…</p>}
                    {error && <p className={styles.errorText}>{error}</p>}

                    {info && (
                        <div className={styles.statusGrid}>
                            <StatusPill
                                ok={info.deployment_mode === 'self_host'}
                                label={`Mode: ${info.deployment_mode}`}
                            />
                            <StatusPill ok={llmReady} label={llmReady ? 'LLM ready' : 'LLM not configured'} />
                            <StatusPill
                                ok={compute.sklearn_available}
                                label={compute.sklearn_available ? 'ML compute ready' : 'ML libs missing'}
                            />
                            <StatusPill
                                ok={compute.duckdb_available}
                                label={compute.duckdb_available ? 'DuckDB pushdown' : 'DuckDB unavailable'}
                            />
                        </div>
                    )}

                    <section className={styles.section}>
                        <h2 className={styles.sectionTitle}>
                            <IconDatabase size={18} />
                            One-command Docker (recommended)
                        </h2>
                        <p className={styles.sectionDesc}>
                            Web UI at <strong>http://localhost:3000</strong> — upload files, chat, train models.
                            All compute runs in your containers using your RAM/CPU.
                        </p>
                        <pre className={styles.codeBlock}>
                            {`cd neural-analyst\ndocker compose up --build\n\n# Optional: background jobs + Redis\n# .\\scripts\\self_host.ps1 -Celery -Detach`}
                        </pre>
                    </section>

                    <section className={styles.section}>
                        <h2 className={styles.sectionTitle}>
                            <IconSparkles size={18} />
                            LLM provider (BYOK)
                        </h2>
                        <p className={styles.sectionDesc}>
                            Keys live in <code>ai-data-analyst/backend/.env</code> — never in the browser.
                            Talk-to-Data works without any key; chat narration needs a provider below.
                        </p>

                        <div className={styles.providers}>
                            {providerGuides.map((p) => (
                                <button
                                    key={p.id}
                                    type="button"
                                    className={`${styles.providerBtn} ${selectedProvider === p.id ? styles.active : ''}`}
                                    onClick={() => setSelectedProvider(p.id)}
                                >
                                    {p.name}
                                    {p.id === 'ollama' && <span className={styles.freeTag}>Free</span>}
                                </button>
                            ))}
                        </div>

                        <pre className={styles.codeBlock}>{envSnippet}</pre>

                        <p className={styles.hint}>
                            Get keys from{' '}
                            <a href={provider.docs} target="_blank" rel="noopener noreferrer">
                                {provider.name}
                            </a>
                            . Full reference: <code>docs/API_KEYS.md</code>
                        </p>

                        {info && (
                            <div className={styles.keyStatus}>
                                <span>OpenAI: {keys.openai ? 'set' : 'missing'}</span>
                                <span>Anthropic: {keys.anthropic ? 'set' : 'missing'}</span>
                                <span>Gemini: {keys.gemini ? 'set' : 'missing'}</span>
                                <span>Current model: {info.llm_model}</span>
                            </div>
                        )}
                    </section>

                    <section className={styles.section}>
                        <h2 className={styles.sectionTitle}>Why web UI, not CLI?</h2>
                        <p className={styles.sectionDesc}>
                            Upload, preview tables, chat with grounded answers, and compare ML models —
                            all need a UI. Docker gives one-command setup; the browser is where you work.
                        </p>
                    </section>

                    <div className={styles.actions}>
                        <Link href="/app/analysis/new" className={styles.primaryBtn}>
                            Open analysis chat
                            <IconArrowRight size={18} />
                        </Link>
                        <Link href="/app/dashboard" className={styles.secondaryBtn}>
                            Dashboard
                        </Link>
                    </div>

                    <div className={styles.benefits}>
                        <div className={styles.benefit}>
                            <IconShield size={20} className={styles.benefitIcon} />
                            <div>
                                <strong>Your compute, your RAM</strong>
                                <p>pandas, sklearn, AutoML, DuckDB — all on your hardware</p>
                            </div>
                        </div>
                        <div className={styles.benefit}>
                            <IconKey size={20} className={styles.benefitIcon} />
                            <div>
                                <strong>Your API keys</strong>
                                <p>Pay providers directly; we never bill for LLM usage in self-host</p>
                            </div>
                        </div>
                    </div>

                    <p className={styles.altPrompt}>
                        Want managed hosting instead?{' '}
                        <Link href="/pricing" className={styles.altLink}>View SaaS plans</Link>
                    </p>
                </motion.div>
            </div>
        </main>
    );
}