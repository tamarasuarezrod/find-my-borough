'use client'

import { useState } from 'react'
import CircleRating from '@/components/circle-rating'
import { useSession } from 'next-auth/react'
import {
  useCommunityScores,
  useSubmitCommunityRatings,
} from '@/services/community-rating'
import { BoroughScore } from '@/types/borough'

import {
  Users,
  Leaf,
  ShieldCheck,
  Sparkles,
  Bus,
  Wifi,
  Baby,
  Footprints,
  Smile,
  Sprout,
} from 'lucide-react'
import { Loader } from '@/components/loader'
import { showErrorToast } from '@/lib/utils'
import toast from 'react-hot-toast'

type CommunityRatingsProps = {
  boroughSlug: string
}

const featureIcons: Record<string, React.ReactNode> = {
  diversity: <Users size={16} />,
  green: <Leaf size={16} />,
  safety: <ShieldCheck size={16} />,
  cleanliness: <Sparkles size={16} />,
  transport: <Bus size={16} />,
  internet: <Wifi size={16} />,
  family: <Baby size={16} />,
  walkability: <Footprints size={16} />,
  vibe: <Smile size={16} />,
  openness: <Sprout size={16} />,
}

export default function CommunityRatings({
  boroughSlug,
}: CommunityRatingsProps) {
  const { status } = useSession()
  const { data: scores, isLoading: isLoadingBoroughScores } =
    useCommunityScores(boroughSlug as string)
  const { mutateAsync: submitRatings, isPending: isSendingScore } =
    useSubmitCommunityRatings({
      onSuccess: () => {
        toast.success('Thanks for your vote!')
        setIsVoting(false)
        setVotes({})
      },
      onError: () => {
        showErrorToast('Failed to submit your vote')
      },
    })

  const [isVoting, setIsVoting] = useState(false)
  const [votes, setVotes] = useState<BoroughScore['ratings']>({})

  const isLoading = isLoadingBoroughScores || isSendingScore

  const handleVote = (featureId: string, score: number) => {
    setVotes((prev) => ({ ...prev, [featureId]: score }))
  }

  const startVoting = () => {
    if (status === 'unauthenticated') {
      showErrorToast("Please log in to vote, we'd love to hear your opinion!")
    } else {
      setIsVoting(true)
      setVotes({})
    }
  }

  const submit = async () => {
    await submitRatings({ borough: boroughSlug, ratings: votes })
  }

  return (
    <div className="mt-12">
      {isLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/70">
          <Loader />
        </div>
      )}
      <h2 className="mb-4 flex items-center gap-3 text-xl font-semibold text-white">
        What people are saying
      </h2>
      <div className="grid grid-cols-1 gap-y-6 text-sm text-white sm:grid-cols-3 sm:gap-x-4">
        {(scores || []).map(({ label, feature }) => {
          const currentScore =
            scores?.find((s) => s.feature === feature)?.score || 0
          const score = isVoting ? (votes[feature] ?? 0) : currentScore

          return (
            <div key={feature}>
              <p className="mb-1 flex items-center gap-2">
                {featureIcons[feature]} {label}
              </p>
              <CircleRating
                score={score}
                editable={isVoting && !isLoading}
                onChange={(val) => handleVote(feature, val)}
              />
            </div>
          )
        })}
      </div>

      {!isLoading && (
        <div className="mt-6 flex justify-end">
          {isVoting ? (
            <button
              onClick={submit}
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
      )}
    </div>
  )
}
