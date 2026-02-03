import { IconCheck, IconAlert, IconX, IconShield } from '@/components/icons';

interface ReadinessBadgeProps {
    level: string;
    score?: number;
    size?: 'sm' | 'md' | 'lg';
}

export default function ReadinessBadge({ level, score, size = 'md' }: ReadinessBadgeProps) {
    const getBadgeConfig = (level: string) => {
        switch (level) {
            case 'READY':
                return {
                    color: 'text-green-500',
                    bgColor: 'bg-green-500/10',
                    borderColor: 'border-green-500/20',
                    label: 'Ready for Deployment',
                    icon: IconCheck
                };
            case 'PARTIALLY_READY':
                return {
                    color: 'text-yellow-500',
                    bgColor: 'bg-yellow-500/10',
                    borderColor: 'border-yellow-500/20',
                    label: 'Partially Ready',
                    icon: IconAlert
                };
            case 'UNSAFE':
                return {
                    color: 'text-red-500',
                    bgColor: 'bg-red-500/10',
                    borderColor: 'border-red-500/20',
                    label: 'Unsafe / Risky',
                    icon: IconX
                };
            case 'BLOCKED':
                return {
                    color: 'text-red-600',
                    bgColor: 'bg-red-600/10',
                    borderColor: 'border-red-600/20',
                    label: 'Deployment Blocked',
                    icon: IconShield
                };
            default:
                return {
                    color: 'text-gray-500',
                    bgColor: 'bg-gray-500/10',
                    borderColor: 'border-gray-500/20',
                    label: 'Unknown Status',
                    icon: IconShield
                };
        }
    };

    const config = getBadgeConfig(level);
    const Icon = config.icon;

    const sizeClasses = {
        sm: 'text-xs px-2 py-1',
        md: 'text-sm px-3 py-1.5',
        lg: 'text-base px-4 py-2'
    };

    return (
        <div className={`inline-flex items-center gap-2 rounded-full border ${config.bgColor} ${config.borderColor} ${config.color} ${sizeClasses[size]}`}>
            <Icon size={size === 'lg' ? 20 : 16} />
            <span className="font-semibold">{config.label}</span>
            {score !== undefined && (
                <>
                    <span className="w-1 h-1 rounded-full bg-current opacity-50" />
                    <span>{(score * 100).toFixed(0)}%</span>
                </>
            )}
        </div>
    );
}
