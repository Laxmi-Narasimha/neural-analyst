'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { IconMail, IconLock, IconEye, IconEyeOff, IconArrowRight, IconKey } from '@/components/icons';
import api from '@/lib/api';
import styles from './page.module.css';

export default function LoginPage() {
    const router = useRouter();
    const [showPassword, setShowPassword] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setErrorMessage(null);
        try {
            await api.login(email, password);
            router.replace('/app/dashboard');
        } catch (err) {
            setErrorMessage(err?.message || 'Login failed');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                {/* Left Side - Form */}
                <motion.div
                    className={styles.formSection}
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className={styles.formWrapper}>
                        <div className={styles.formHeader}>
                            <h1 className={styles.title}>Welcome back</h1>
                            <p className={styles.subtitle}>
                                Sign in to access your AI data analyst
                            </p>
                        </div>

                        {errorMessage && (
                            <div style={{ color: 'var(--error-500)', fontSize: 14, marginBottom: 12 }}>
                                {errorMessage}
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className={styles.form}>
                            <div className={styles.inputGroup}>
                                <label htmlFor="email" className={styles.label}>Email</label>
                                <div className={styles.inputWrapper}>
                                    <IconMail size={20} className={styles.inputIcon} />
                                    <input
                                        type="email"
                                        id="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="you@company.com"
                                        required
                                        className={styles.input}
                                    />
                                </div>
                            </div>

                            <div className={styles.inputGroup}>
                                <div className={styles.labelRow}>
                                    <label htmlFor="password" className={styles.label}>Password</label>
                                    <Link href="/forgot-password" className={styles.forgotLink}>
                                        Forgot password?
                                    </Link>
                                </div>
                                <div className={styles.inputWrapper}>
                                    <IconLock size={20} className={styles.inputIcon} />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        id="password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="Enter your password"
                                        required
                                        className={styles.input}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className={styles.togglePassword}
                                        aria-label={showPassword ? 'Hide password' : 'Show password'}
                                    >
                                        {showPassword ? <IconEyeOff size={20} /> : <IconEye size={20} />}
                                    </button>
                                </div>
                            </div>

                            <button
                                type="submit"
                                className={styles.submitBtn}
                                disabled={isLoading}
                            >
                                {isLoading ? (
                                    <span className={styles.spinner}></span>
                                ) : (
                                    <>
                                        Sign In
                                        <IconArrowRight size={20} />
                                    </>
                                )}
                            </button>
                        </form>

                        <div className={styles.divider}>
                            <span>or</span>
                        </div>

                        <Link href="/setup-keys" className={styles.byokBtn}>
                            <IconKey size={20} />
                            Continue with Your Own API Keys
                        </Link>

                        <p className={styles.registerPrompt}>
                            Don't have an account?{' '}
                            <Link href="/register" className={styles.registerLink}>
                                Create one free
                            </Link>
                        </p>
                    </div>
                </motion.div>

                {/* Right Side - Visual */}
                <motion.div
                    className={styles.visualSection}
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <div className={styles.visualContent}>
                        <div className={styles.visualGlow}></div>
                        <h2 className={styles.visualTitle}>
                            Analyze data
                            <span className={styles.gradient}> 24/7</span>
                        </h2>
                        <p className={styles.visualDesc}>
                            Your AI analyst works while you sleep. Process millions of rows,
                            build ML models, and generate insights — automatically.
                        </p>
                        <div className={styles.visualStats}>
                            <div className={styles.visualStat}>
                                <span className={styles.statValue}>244+</span>
                                <span className={styles.statLabel}>Features</span>
                            </div>
                            <div className={styles.visualStat}>
                                <span className={styles.statValue}>3.5M</span>
                                <span className={styles.statLabel}>Rows/Hour</span>
                            </div>
                            <div className={styles.visualStat}>
                                <span className={styles.statValue}>99.9%</span>
                                <span className={styles.statLabel}>Uptime</span>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </main>
    );
}
