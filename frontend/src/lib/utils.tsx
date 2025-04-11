import {
  ArrowUp,
  ArrowDown,
  ArrowDownUp,
  House,
  Users,
  PartyPopper,
  PoundSterling,
  MapPinned,
} from 'lucide-react'

export function toTitleCase(str: string): string {
  return str.replace(/\b\w/g, (char) => char.toUpperCase())
}

export function getYouthIndicator(score: number | undefined) {
  if (!score) return null

  if (score >= 0.6) {
    return {
      label: 'Youthful',
      icon: <PartyPopper size={14} />,
    }
  } else if (score >= 0.3) {
    return {
      label: 'Mixed ages',
      icon: <Users size={14} />,
    }
  } else {
    return {
      label: 'Residential',
      icon: <House size={14} />,
    }
  }
}

export function getRentIndicator(score: number | undefined) {
  if (!score) return null

  if (score >= 0.66) {
    return {
      icon: <PoundSterling size={14} className="text-green-400" />,
      color: 'text-green-400',
      label: 'Affordable',
    }
  } else if (score >= 0.33) {
    return {
      icon: <PoundSterling size={14} className="text-yellow-400" />,
      color: 'text-yellow-400',
      label: 'Moderate',
    }
  } else {
    return {
      icon: <PoundSterling size={14} className="text-red-400" />,
      color: 'text-red-400',
      label: 'High rent',
    }
  }
}

export function getCrimeIndicator(score: number | undefined) {
  if (!score) return null

  const crime = 1 - score

  if (crime >= 0.6) {
    return {
      icon: <ArrowUp size={14} className="text-red-400" />,
      color: 'text-red-400',
      label: 'High crime',
      shortLabel: 'Crime',
    }
  } else if (crime >= 0.3) {
    return {
      icon: <ArrowDownUp size={14} className="text-yellow-400" />,
      color: 'text-yellow-400',
      label: 'Medium crime',
      shortLabel: 'Crime',
    }
  } else {
    return {
      icon: <ArrowDown size={14} className="text-green-400" />,
      color: 'text-green-400',
      label: 'Low crime',
      shortLabel: 'Crime',
    }
  }
}

export function getCentralityIndicator(score: number | undefined) {
  if (score === undefined || score === null) return null

  if (score >= 0.6) {
    return {
      icon: <MapPinned size={14} />,
      color: 'text-red-400',
      label: 'Central',
    }
  } else if (score >= 0.3) {
    return {
      icon: <MapPinned size={14} />,
      color: 'text-yellow-400',
      label: 'Medium',
    }
  } else {
    return {
      icon: <MapPinned size={14} />,
      color: 'text-green-400',
      label: 'Outer',
    }
  }
}
