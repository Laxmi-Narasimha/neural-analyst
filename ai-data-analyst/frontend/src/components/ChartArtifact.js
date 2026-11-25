'use client';

import dynamic from 'next/dynamic';
import { artifactLabel } from '@/lib/labels';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function extractPlotSpec(artifact) {
    const preview = artifact?.preview || {};
    const spec = preview?.spec?.spec || preview?.spec || {};
    if (spec?.data && spec?.layout) return spec;
    return null;
}

export default function ChartArtifact({ artifact, title, compact = false }) {
    const plotSpec = extractPlotSpec(artifact);
    if (!plotSpec) return null;

    const chartTitle = title || artifactLabel(artifact?.name) || previewTitle(artifact) || 'Chart';

    return (
        <div style={{
            marginTop: compact ? '0.5rem' : '0.75rem',
            border: '1px solid rgba(139, 92, 246, 0.2)',
            borderRadius: '12px',
            overflow: 'hidden',
            background: '#fff',
        }}>
            <div style={{
                padding: '0.45rem 0.75rem',
                fontSize: '0.72rem',
                fontWeight: 700,
                color: '#5b21b6',
                borderBottom: '1px solid rgba(139, 92, 246, 0.12)',
            }}>
                {chartTitle}
            </div>
            <Plot
                data={plotSpec.data}
                layout={{
                    ...plotSpec.layout,
                    autosize: true,
                    margin: { l: 40, r: 20, t: 40, b: 40 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    height: compact ? 280 : 360,
                }}
                config={{ displayModeBar: !compact, responsive: true }}
                style={{ width: '100%' }}
                useResizeHandler
            />
        </div>
    );
}

function previewTitle(artifact) {
    const t = artifact?.preview?.spec?.title;
    return typeof t === 'string' ? t : null;
}

export function extractChartsFromAgentActions(agentActions) {
    if (!Array.isArray(agentActions)) return [];
    const charts = [];
    for (const action of agentActions) {
        for (const art of action?.artifacts || []) {
            const type = String(art?.artifact_type || '').toLowerCase();
            if (type === 'chart') charts.push(art);
        }
    }
    return charts;
}