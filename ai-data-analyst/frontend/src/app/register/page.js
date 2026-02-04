'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { IconMail, IconLock, IconUser, IconEye, IconEyeOff, IconArrowRight, IconCheck } from '@/components/icons';
import api from '@/lib/api';
import styles from '../login/page.module.css';

export default function RegisterPage() {
    const router = useRouter();
    const [showPassword, setShowPassword] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
    });
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState(null);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setErrorMessage(null);
        try {
            await api.register(formData.email, formData.password, formData.name, 'analyst');
            await api.login(formData.email, formData.password);
            router.replace('/app/dashboard');
        } catch (err) {
            setErrorMessage(err?.message || 'Registration failed');
        } finally {
            setIsLoading(false);
        }
    };

    const passwordStrength = () => {
        const p = formData.password;
        if (p.length === 0) return 0;
        if (p.length < 6) return 1;
        if (p.length < 8) return 2;
        if (p.length >= 8 && /[A-Z]/.test(p) && /[0-9]/.test(p)) return 4;
        return 3;
    };

    const strengthLabels = ['', 'Weak', 'Fair', 'Good', 'Strong'];
    const strengthColors = ['', 'var(--error-400)', 'var(--warning-400)', 'var(--glow-400)', 'var(--glow-500)'];

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
                            <h1 className={styles.title}>Create your account</h1>
                            <p className={styles.subtitle}>
                                Start analyzing data in minutes — free forever
                            </p>
                        </div>

                        {errorMessage && (
                            <div style={{ color: 'var(--error-500)', fontSize: 14, marginBottom: 12 }}>
                                {errorMessage}
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className={styles.form}>
                            <div className={styles.inputGroup}>
                                <label htmlFor="name" className={styles.label}>Full Name</label>
                                <div className={styles.inputWrapper}>
                                    <IconUser size={20} className={styles.inputIcon} />
                                    <input
                                        type="text"
                                        id="name"
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                        placeholder="Your name"
                                        required
                                        className={styles.input}
                                    />
                                </div>
                            </div>

                            <div className={styles.inputGroup}>
                                <label htmlFor="email" className={styles.label}>Work Email</label>
                                <div className={styles.inputWrapper}>
                                    <IconMail size={20} className={styles.inputIcon} />
                                    <input
                                        type="email"
                                        id="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        placeholder="you@company.com"
                                        required
                                        className={styles.input}
                                    />
                                </div>
                            </div>

                            <div className={styles.inputGroup}>
                                <label htmlFor="password" className={styles.label}>Password</label>
                                <div className={styles.inputWrapper}>
                                    <IconLock size={20} className={styles.inputIcon} />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        id="password"
                                        name="password"
                                        value={formData.password}
                                        onChange={handleChange}
                                        placeholder="Create a strong password"
                                        required
                                        minLength={8}
                                        className={styles.input}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className={styles.togglePassword}
                                    >
                                        {showPassword ? <IconEyeOff size={20} /> : <IconEye size={20} />}
                                    </button>
                                </div>
                                {formData.password && (
                                    <div style={{ marginTop: '8px' }}>
                                        <div style={{
                                            display: 'flex',
                                            gap: '4px',
                                            marginBottom: '4px'
                                        }}>
                                            {[1, 2, 3, 4].map((level) => (
                                                <div
                                                    key={level}
                                                    style={{
                                                        flex: 1,
                                                        height: '3px',
                                                        borderRadius: '2px',
                                                        background: level <= passwordStrength()
                                                            ? strengthColors[passwordStrength()]
                                                            : 'rgba(148, 163, 184, 0.2)',
                                                        transition: 'background 0.3s',
                                                    }}
                                                />
                                            ))}
                                        </div>
                                        <span style={{
                                            fontSize: '12px',
                                            color: strengthColors[passwordStrength()]
                                        }}>
                                            {strengthLabels[passwordStrength()]}
                                        </span>
                                    </div>
                                )}
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
                                        Create Account
                                        <IconArrowRight size={20} />
                                    </>
                                )}
                            </button>
                        </form>

                        <p style={{
                            fontSize: '12px',
                            color: 'var(--text-500)',
                            textAlign: 'center',
                            marginTop: 'var(--space-4)'
                        }}>
                            By signing up, you agree to our{' '}
                            <Link href="/terms" style={{ color: 'var(--primary-400)' }}>Terms</Link>
                            {' '}and{' '}
                            <Link href="/privacy" style={{ color: 'var(--primary-400)' }}>Privacy Policy</Link>
                        </p>

                        <p className={styles.registerPrompt}>
                            Already have an account?{' '}
                            <Link href="/login" className={styles.registerLink}>
                                Sign in
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
                            Included in
                            <span className={styles.gradient}> Free</span>
                        </h2>
                        <ul style={{
                            listStyle: 'none',
                            textAlign: 'left',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 'var(--space-4)'
                        }}>
                            {[
                                '100 queries per month',
                                'Basic EDA & statistics',
                                '10MB dataset upload',
                                '5 visualizations per day',
                                'Community support',
                            ].map((feature) => (
                                <li key={feature} style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 'var(--space-3)',
                                    color: 'var(--text-300)',
                                }}>
                                    <span style={{
                                        color: 'var(--glow-400)',
                                        flexShrink: 0,
                                    }}>
                                        <IconCheck size={18} />
                                    </span>
                                    {feature}
                                </li>
                            ))}
                        </ul>
                    </div>
                </motion.div>
            </div>
        </main>
    );
}
