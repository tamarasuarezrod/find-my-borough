'use client'

import { ThumbsUp, ThumbsDown } from 'lucide-react'

interface FeedbackButtonsProps {
  boroughName: string
  onSendFeedback: (borough: string, feedback: string) => void
  variant?: 'icon-only' | 'text'
}

export default function FeedbackButtons({
  boroughName,
  onSendFeedback,
  variant = 'text',
}: FeedbackButtonsProps) {
  if (variant === 'icon-only') {
    return (
      <div className="absolute right-3 top-3 z-10 flex gap-2">
        <button
          onClick={() => onSendFeedback(boroughName, 'like')}
          className="rounded-full bg-zinc-800/80 p-2 text-gray-300 transition hover:bg-green-500 hover:text-white"
        >
          <ThumbsUp className="h-4 w-4" strokeWidth={1.5} />
        </button>
        <button
          onClick={() => onSendFeedback(boroughName, 'dislike')}
          className="rounded-full bg-zinc-800/80 p-2 text-gray-300 transition hover:bg-red-500 hover:text-white"
        >
          <ThumbsDown className="h-4 w-4" strokeWidth={1.5} />
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 text-sm text-gray-500">
      <button
        onClick={() => onSendFeedback(boroughName, 'like')}
        className="flex items-center gap-1 text-gray-400 transition hover:text-green-400"
      >
        <ThumbsUp className="h-4 w-4" strokeWidth={1.5} />
        Yes
      </button>
      <button
        onClick={() => onSendFeedback(boroughName, 'dislike')}
        className="flex items-center gap-1 text-gray-400 transition hover:text-red-400"
      >
        <ThumbsDown className="h-4 w-4" strokeWidth={1.5} />
        No
      </button>
    </div>
  )
}
