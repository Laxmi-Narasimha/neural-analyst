'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { IconMail, IconUser, IconChat, IconArrowRight, IconBuilding, IconGlobe } from '@/components/icons';
import styles from './page.module.css';

export default function ContactPage() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        company: '',
        type: 'general',
        message: '',
    });
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isSubmitted, setIsSubmitted] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        setTimeout(() => {
            setIsSubmitting(false);
            setIsSubmitted(true);
        }, 1500);
    };

    return (
        <main className={styles.main}>
            <div className={styles.container}>
                {/* Header */}
                <motion.div
                    className={styles.header}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <h1 className={styles.title}>
                        Get in <span className={styles.gradient}>Touch</span>
                    </h1>
                    <p className={styles.subtitle}>
                        Have questions? We'd love to hear from you. Send us a message
                        and we'll respond as soon as possible.
                    </p>
                </motion.div>

                <div className={styles.grid}>
                    {/* Contact Form */}
                    <motion.div
                        className={styles.formCard}
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                    >
                        {isSubmitted ? (
                            <div className={styles.successMessage}>
                                <div className={styles.successIcon}>
                                    <IconChat size={32} />
                                </div>
                                <h2>Message Sent!</h2>
                                <p>Thank you for reaching out. We'll get back to you within 24 hours.</p>
                            </div>
                        ) : (
                            <form onSubmit={handleSubmit} className={styles.form}>
                                <div className={styles.formRow}>
                                    <div className={styles.formGroup}>
                                        <label className={styles.label}>Your Name</label>
                                        <div className={styles.inputWrapper}>
                                            <IconUser size={18} className={styles.inputIcon} />
                                            <input
                                                type="text"
                                                name="name"
                                                value={formData.name}
                                                onChange={handleChange}
                                                placeholder="John Doe"
                                                required
                                                className={styles.input}
                                            />
                                        </div>
                                    </div>
                                    <div className={styles.formGroup}>
                                        <label className={styles.label}>Email Address</label>
                                        <div className={styles.inputWrapper}>
                                            <IconMail size={18} className={styles.inputIcon} />
                                            <input
                                                type="email"
                                                name="email"
                                                value={formData.email}
                                                onChange={handleChange}
                                                placeholder="john@company.com"
                                                required
                                                className={styles.input}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className={styles.formGroup}>
                                    <label className={styles.label}>Company (Optional)</label>
                                    <div className={styles.inputWrapper}>
                                        <IconBuilding size={18} className={styles.inputIcon} />
                                        <input
                                            type="text"
                                            name="company"
                                            value={formData.company}
                                            onChange={handleChange}
                                            placeholder="Acme Inc."
                                            className={styles.input}
                                        />
                                    </div>
                                </div>

                                <div className={styles.formGroup}>
                                    <label className={styles.label}>Inquiry Type</label>
                                    <select
                                        name="type"
                                        value={formData.type}
                                        onChange={handleChange}
                                        className={styles.select}
                                    >
                                        <option value="general">General Inquiry</option>
                                        <option value="sales">Sales</option>
                                        <option value="support">Technical Support</option>
                                        <option value="enterprise">Enterprise Plan</option>
                                        <option value="partnership">Partnership</option>
                                    </select>
                                </div>

                                <div className={styles.formGroup}>
                                    <label className={styles.label}>Message</label>
                                    <textarea
                                        name="message"
                                        value={formData.message}
                                        onChange={handleChange}
                                        placeholder="Tell us how we can help..."
                                        required
                                        rows={5}
                                        className={styles.textarea}
                                    />
                                </div>

                                <button
                                    type="submit"
                                    className={styles.submitBtn}
                                    disabled={isSubmitting}
                                >
                                    {isSubmitting ? (
                                        <span className={styles.spinner}></span>
                                    ) : (
                                        <>
                                            Send Message
                                            <IconArrowRight size={18} />
                                        </>
                                    )}
                                </button>
                            </form>
                        )}
                    </motion.div>

                    {/* Contact Info */}
                    <motion.div
                        className={styles.infoSection}
                        initial={{ opacity: 0, x: 30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                    >
                        <div className={styles.infoCard}>
                            <IconMail size={24} className={styles.infoIcon} />
                            <h3>Email Us</h3>
                            <p>support@neuralanalyst.ai</p>
                        </div>
                        <div className={styles.infoCard}>
                            <IconChat size={24} className={styles.infoIcon} />
                            <h3>Live Chat</h3>
                            <p>Available Mon-Fri, 9am-6pm EST</p>
                        </div>
                        <div className={styles.infoCard}>
                            <IconGlobe size={24} className={styles.infoIcon} />
                            <h3>Community</h3>
                            <p>Join our Discord community</p>
                        </div>
                    </motion.div>
                </div>
            </div>
        </main>
    );
}
