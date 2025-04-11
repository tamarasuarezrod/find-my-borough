'use client'

import { useState } from 'react'
import toast from 'react-hot-toast'
import CircleRating from '@/components/circle-rating'
import { useSession } from 'next-auth/react'
import {
  useCommunityScores,
  useSubmitCommunityRatings,
} from '@/services/community-rating'
import { Score } from '@/types/borough'

type CommunityRatingsProps = {
  boroughSlug: string
}

export default function CommunityRatings({
  boroughSlug,
}: CommunityRatingsProps) {
  const { status } = useSession()
  const { data: scores, refetch } = useCommunityScores(boroughSlug) as {
    data: Score[] | undefined
    refetch: () => void
  }
  const { mutateAsync: submitRatings } = useSubmitCommunityRatings()

  const [isVoting, setIsVoting] = useState(false)
  const [votes, setVotes] = useState<Record<string, number>>({})

  const handleVote = (featureId: string, score: number) => {
    setVotes((prev) => ({ ...prev, [featureId]: score }))
  }

  const startVoting = () => {
    if (status === 'unauthenticated') {
      toast.error("Please log in to vote, we'd love to hear your opinion!")
    } else {
      setIsVoting(true)
      const defaultVotes = Object.fromEntries(
        (scores || []).map(({ feature }: Score) => [feature, 0]),
      )
      setVotes(defaultVotes)
    }
  }

  const submit = async () => {
    try {
      await submitRatings({ borough: boroughSlug, ratings: votes })
      toast.success('Thanks for your vote!')
      setIsVoting(false)
      setVotes({})
      refetch()
    } catch {
      toast.error('Failed to submit your vote')
    }
  }

  return (
    <div className="mt-12">
      <h2 className="mb-4 text-xl font-semibold text-white">
        What people are saying
      </h2>

      <div className="grid grid-cols-2 gap-x-4 gap-y-6 text-sm text-white sm:grid-cols-3">
        {(scores || []).map(({ label, feature }) => {
          const currentScore =
            scores?.find((s) => s.feature === feature)?.score || 0
          const score = isVoting ? (votes[feature] ?? 0) : currentScore

          return (
            <div key={feature}>
              <p className="mb-1 flex items-center gap-2">{label}</p>
              <CircleRating
                score={score}
                editable={isVoting}
                onChange={(val) => handleVote(feature, val)}
              />
            </div>
          )
        })}
      </div>

      <div className="mt-6 flex justify-end">
        {isVoting ? (
          <button onClick={submit} className="text-sm text-green-400 underline">
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
