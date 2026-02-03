'use client';

import { useState } from 'react';
import { IconUpload, IconShield, IconBrain } from '@/components/icons';
import QualityDashboard from '@/components/quality/QualityDashboard';

export default function DataQualityPage() {
    const [activeTab, setActiveTab] = useState('dashboard');

    return (
        <div className="p-6 max-w-7xl mx-auto">
            <header className="mb-8">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-500 to-purple-600 bg-clip-text text-transparent flex items-center gap-3">
                    <IconShield size={32} className="text-cyan-500" />
                    Data Adequacy Analyst
                </h1>
                <p className="text-gray-400 mt-2">
                    Evaluate your data's readiness for AI applications with comprehensive quality scoring.
                </p>
            </header>

            <div className="bg-[#1e1e1e] rounded-xl border border-gray-800 p-6 min-h-[600px]">
                <QualityDashboard />
            </div>
        </div>
    );
}
