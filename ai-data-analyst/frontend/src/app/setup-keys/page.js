'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { IconKey, IconShield, IconCheck, IconArrowRight, IconEye, IconEyeOff } from '@/components/icons';
import styles from './page.module.css';

const providers = [
    {
        id: 'openai',
        name: 'OpenAI',
        placeholder: 'sk-proj-...',
        docs: 'https://platform.openai.com/api-keys',
        models: ['GPT-4o', 'GPT-4 Turbo', 'GPT-3.5 Turbo'],
    },
    {
        id: 'gemini',
        name: 'Google Gemini',
        placeholder: 'AI...',
        docs: 'https://aistudio.google.com/app/apikey',
        models: ['Gemini 1.5 Pro', 'Gemini 1.5 Flash', 'Gemini Pro'],
    },
    {
        id: 'anthropic',
        name: 'Anthropic',
        placeholder: 'sk-ant-...',
        docs: 'https://console.anthropic.com/settings/keys',
        models: ['Claude 3.5 Sonnet', 'Claude 3 Opus', 'Claude 3 Haiku'],
    },
];

export default function SetupKeysPage() {
    const [selectedProvider, setSelectedProvider] = useState('openai');
    const [apiKey, setApiKey] = useState('');
    const [showKey, setShowKey] = useState(false);
    const [isValidating, setIsValidating] = useState(false);
    const [isValid, setIsValid] = useState(null);

    const currentProvider = providers.find(p => p.id === selectedProvider);

    const validateKey = async () => {
        if (!apiKey) return;
        setIsValidating(true);
        // Simulate API validation
        setTimeout(() => {
            setIsValidating(false);
            setIsValid(apiKey.length > 10);
        }, 1500);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (isValid) {
            window.location.href = '/app/dashboard';
        }
    };

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className={styles.content}
                >
                    {/* Header */}
                    <div className={styles.header}>
                        <div className={styles.iconWrapper}>
                            <IconKey size={32} />
                        </div>
                        <h1 className={styles.title}>
                            <span className={styles.gradient}>BYOK</span> Setup
                        </h1>
                        <p className={styles.subtitle}>
                            Use your own API keys for complete privacy and control.
                            You only pay for what you use through your provider.
                        </p>
                    </div>

                    {/* Provider Selection */}
                    <div className={styles.providers}>
                        {providers.map((provider) => (
                            <button
                                key={provider.id}
                                className={`${styles.providerBtn} ${selectedProvider === provider.id ? styles.active : ''}`}
                                onClick={() => {
                                    setSelectedProvider(provider.id);
                                    setApiKey('');
                                    setIsValid(null);
                                }}
                            >
                                {provider.name}
                            </button>
                        ))}
                    </div>

                    {/* Key Input */}
                    <form onSubmit={handleSubmit} className={styles.form}>
                        <div className={styles.inputSection}>
                            <label className={styles.label}>
                                {currentProvider.name} API Key
                            </label>
                            <div className={styles.inputWrapper}>
                                <input
                                    type={showKey ? 'text' : 'password'}
                                    value={apiKey}
                                    onChange={(e) => {
                                        setApiKey(e.target.value);
                                        setIsValid(null);
                                    }}
                                    placeholder={currentProvider.placeholder}
                                    className={`${styles.input} ${isValid === true ? styles.valid : ''} ${isValid === false ? styles.invalid : ''}`}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowKey(!showKey)}
                                    className={styles.toggleBtn}
                                >
                                    {showKey ? <IconEyeOff size={20} /> : <IconEye size={20} />}
                                </button>
                            </div>
                            <p className={styles.hint}>
                                Get your API key from{' '}
                                <a href={currentProvider.docs} target="_blank" rel="noopener noreferrer">
                                    {currentProvider.name} Console
                                </a>
                            </p>
                        </div>

                        {/* Validate Button */}
                        <button
                            type="button"
                            onClick={validateKey}
                            disabled={!apiKey || isValidating}
                            className={styles.validateBtn}
                        >
                            {isValidating ? (
                                <span className={styles.spinner}></span>
                            ) : isValid === true ? (
                                <>
                                    <IconCheck size={20} />
                                    Key Validated
                                </>
                            ) : (
                                'Validate Key'
                            )}
                        </button>

                        {/* Models Available */}
                        {isValid && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                className={styles.modelsSection}
                            >
                                <h3 className={styles.modelsTitle}>Available Models</h3>
                                <div className={styles.modelsList}>
                                    {currentProvider.models.map((model) => (
                                        <div key={model} className={styles.modelItem}>
                                            <IconCheck size={16} className={styles.modelCheck} />
                                            {model}
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}

                        {/* Submit */}
                        <button
                            type="submit"
                            disabled={!isValid}
                            className={styles.submitBtn}
                        >
                            Continue to Dashboard
                            <IconArrowRight size={20} />
                        </button>
                    </form>

                    {/* Benefits */}
                    <div className={styles.benefits}>
                        <div className={styles.benefit}>
                            <IconShield size={20} className={styles.benefitIcon} />
                            <div>
                                <strong>Full Privacy</strong>
                                <p>Your data never leaves your control</p>
                            </div>
                        </div>
                        <div className={styles.benefit}>
                            <IconKey size={20} className={styles.benefitIcon} />
                            <div>
                                <strong>Pay As You Go</strong>
                                <p>Only pay for API calls you make</p>
                            </div>
                        </div>
                    </div>

                    <p className={styles.altPrompt}>
                        Want managed hosting instead?{' '}
                        <Link href="/pricing" className={styles.altLink}>View our plans</Link>
                    </p>
                </motion.div>
            </div>
        </main>
    );
}
