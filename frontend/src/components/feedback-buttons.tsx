'use client'

import { ThumbsUp, ThumbsDown } from 'lucide-react'

interface FeedbackButtonsProps {
  boroughSlug: string
  onSendFeedback: (borough: string, feedback: string) => void
  variant?: 'icon-only' | 'text'
  selected?: boolean
}

export default function FeedbackButtons({
  boroughSlug,
  onSendFeedback,
  variant = 'text',
  selected,
}: FeedbackButtonsProps) {
  const likeSelected = selected === true
  const dislikeSelected = selected === false

  if (variant === 'icon-only') {
    return (
      <div className="absolute right-3 top-3 z-10 flex gap-2">
        <button
          onClick={() => onSendFeedback(boroughSlug, 'like')}
          className={`rounded-full p-2 transition ${
            likeSelected
              ? 'bg-green-500 text-white'
              : 'bg-zinc-800/80 text-gray-300 hover:bg-green-500 hover:text-white'
          }`}
        >
          <ThumbsUp className="h-4 w-4" strokeWidth={1.5} />
        </button>
        <button
          onClick={() => onSendFeedback(boroughSlug, 'dislike')}
          className={`rounded-full p-2 transition ${
            dislikeSelected
              ? 'bg-red-500 text-white'
              : 'bg-zinc-800/80 text-gray-300 hover:bg-red-500 hover:text-white'
          }`}
        >
          <ThumbsDown className="h-4 w-4" strokeWidth={1.5} />
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 text-sm text-gray-500">
      <button
        onClick={() => onSendFeedback(boroughSlug, 'like')}
        className={`flex items-center gap-1 transition ${
          likeSelected ? 'text-green-400' : 'text-gray-400 hover:text-green-400'
        }`}
      >
        <ThumbsUp className="h-4 w-4" strokeWidth={1.5} />
        Yes
      </button>
      <button
        onClick={() => onSendFeedback(boroughSlug, 'dislike')}
        className={`flex items-center gap-1 transition ${
          dislikeSelected ? 'text-red-400' : 'text-gray-400 hover:text-red-400'
        }`}
      >
        <ThumbsDown className="h-4 w-4" strokeWidth={1.5} />
        No
      </button>
    </div>
  )
}
