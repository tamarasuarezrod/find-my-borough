'use client'

import { useState } from 'react'
import toast from 'react-hot-toast'
import CircleRating from '@/components/circle-rating'

type CommunityRatingsProps = {
  ratings: {
    label: string
    icon: React.ReactNode
    score: number
  }[]
}

export default function CommunityRatings({ ratings }: CommunityRatingsProps) {
  const [isVoting, setIsVoting] = useState(false)
  const [votes, setVotes] = useState<number[]>([])

  const handleVote = (index: number, score: number) => {
    const updated = [...votes]
    updated[index] = score
    setVotes(updated)
  }

  const submitVotes = () => {
    toast.success('Thanks for your vote!')
    setIsVoting(false)
    setVotes([])
    // Podés enviar al backend si querés acá
  }

  const startVoting = () => {
    setIsVoting(true)
    setVotes(ratings.map(() => 0)) // inicia todos en 0
  }

  return (
    <div className="mt-12">
      <h2 className="mb-4 text-xl font-semibold text-white">
        What people are saying
      </h2>
      <div className="grid grid-cols-2 gap-x-4 gap-y-6 text-sm text-white sm:grid-cols-3">
        {ratings.map((item, i) => (
          <div key={item.label}>
            <p className="mb-1 flex items-center gap-2">
              {item.icon} {item.label}
            </p>
            <CircleRating
              score={isVoting ? votes[i] : item.score}
              editable={isVoting}
              onChange={(score) => handleVote(i, score)}
            />
          </div>
        ))}
      </div>

      <div className="mt-6 flex justify-end">
        {isVoting ? (
          <button
            onClick={submitVotes}
            className="text-sm text-green-400 underline"
          >
            Submit votes
          </button>
        ) : (
          <button
            onClick={startVoting}
            className="text-sm text-gray-400 underline"
          >
            Add your vote
          </button>
        )}
      </div>
    </div>
  )
}
