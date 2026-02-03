'use client';

import { useState } from 'react';
import { IconSparkles, IconArrowRight, IconCheck } from '@/components/icons';
import styles from './page.module.css';

export default function FeedbackPage() {
    const [feedbackType, setFeedbackType] = useState('feature');
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        // Simulate submission
        setTimeout(() => setSubmitted(true), 1000);
    };

    if (submitted) {
        return (
            <main className={styles.main}>
                <div className={styles.successContainer}>
                    <div className={styles.successIcon}>
                        <IconCheck size={40} />
                    </div>
                    <h1>Thank You!</h1>
                    <p>Your feedback has been submitted. We review all submissions and may reach out for more details.</p>
                    <a href="/" className={styles.homeBtn}>Back to Home</a>
                </div>
            </main>
        );
    }

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <IconSparkles size={32} className={styles.headerIcon} />
                    <h1 className={styles.title}>
                        Share Your <span className={styles.gradient}>Feedback</span>
                    </h1>
                    <p className={styles.subtitle}>
                        Help us build the features you need. We read every submission.
                    </p>
                </div>

                <form onSubmit={handleSubmit} className={styles.form}>
                    {/* Feedback Type */}
                    <div className={styles.typeSelector}>
                        <button
                            type="button"
                            className={`${styles.typeBtn} ${feedbackType === 'feature' ? styles.active : ''}`}
                            onClick={() => setFeedbackType('feature')}
                        >
                            Feature Request
                        </button>
                        <button
                            type="button"
                            className={`${styles.typeBtn} ${feedbackType === 'bug' ? styles.active : ''}`}
                            onClick={() => setFeedbackType('bug')}
                        >
                            Bug Report
                        </button>
                        <button
                            type="button"
                            className={`${styles.typeBtn} ${feedbackType === 'improvement' ? styles.active : ''}`}
                            onClick={() => setFeedbackType('improvement')}
                        >
                            Improvement
                        </button>
                    </div>

                    {/* Title */}
                    <div className={styles.formGroup}>
                        <label>Title</label>
                        <input
                            type="text"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder="Brief summary of your feedback"
                            required
                            className={styles.input}
                        />
                    </div>

                    {/* Description */}
                    <div className={styles.formGroup}>
                        <label>Description</label>
                        <textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Describe your request or issue in detail..."
                            required
                            rows={6}
                            className={styles.textarea}
                        />
                    </div>

                    {/* Email */}
                    <div className={styles.formGroup}>
                        <label>Email (optional)</label>
                        <input
                            type="email"
                            placeholder="For follow-up questions"
                            className={styles.input}
                        />
                    </div>

                    <button type="submit" className={styles.submitBtn}>
                        Submit Feedback
                        <IconArrowRight size={18} />
                    </button>
                </form>
            </div>
        </main>
    );
}
