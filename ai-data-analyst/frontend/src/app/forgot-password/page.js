'use client';

import { useState } from 'react';
import Link from 'next/link';
import { IconMail, IconArrowRight, IconCheck } from '@/components/icons';
import styles from './page.module.css';

export default function ForgotPasswordPage() {
    const [email, setEmail] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        // Simulate API call
        setTimeout(() => {
            setIsLoading(false);
            setSubmitted(true);
        }, 1500);
    };

    if (submitted) {
        return (
            <main className={styles.main}>
                <div className={styles.container}>
                    <div className={styles.successIcon}>
                        <IconCheck size={32} />
                    </div>
                    <h1 className={styles.title}>Check Your Email</h1>
                    <p className={styles.desc}>
                        We've sent a password reset link to <strong>{email}</strong>.
                        The link will expire in 1 hour.
                    </p>
                    <p className={styles.hint}>
                        Didn't receive the email? Check your spam folder or{' '}
                        <button onClick={() => setSubmitted(false)} className={styles.retryLink}>
                            try again
                        </button>.
                    </p>
                    <Link href="/login" className={styles.backLink}>
                        Back to Login
                    </Link>
                </div>
            </main>
        );
    }

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <h1 className={styles.title}>Reset Password</h1>
                <p className={styles.desc}>
                    Enter your email address and we'll send you a link to reset your password.
                </p>

                <form onSubmit={handleSubmit} className={styles.form}>
                    <div className={styles.inputWrapper}>
                        <IconMail size={18} className={styles.inputIcon} />
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Enter your email"
                            required
                            className={styles.input}
                        />
                    </div>

                    <button type="submit" className={styles.submitBtn} disabled={isLoading}>
                        {isLoading ? (
                            <span className={styles.spinner}></span>
                        ) : (
                            <>
                                Send Reset Link
                                <IconArrowRight size={18} />
                            </>
                        )}
                    </button>
                </form>

                <Link href="/login" className={styles.backLink}>
                    <IconArrowRight size={14} className={styles.backIcon} />
                    Back to Login
                </Link>
            </div>
        </main>
    );
}
