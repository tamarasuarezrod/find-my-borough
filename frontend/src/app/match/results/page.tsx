'use client'

import { useEffect, useState } from 'react'
import BoroughCard from '@/components/borough-card'
import {
  getCrimeIndicator,
  getRentIndicator,
  getYouthIndicator,
  toTitleCase,
} from '@/lib/utils'
import { ThumbsDown, ThumbsUp } from 'lucide-react'
import Link from 'next/link'
import { useBoroughsContext } from '@/context/boroughs-context'

type Recommendation = {
  borough: string
  score: number
  norm_rent: number
  norm_crime: number
  norm_youth: number
  norm_centrality: number
}

export default function ResultsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const { boroughs } = useBoroughsContext()

  useEffect(() => {
    const stored = sessionStorage.getItem('recommendations')
    if (stored) {
      const parsed: Recommendation[] = JSON.parse(stored)
      const sorted = [...parsed].sort((a, b) => b.score - a.score)
      setRecommendations(sorted)
    }
  }, [])

  const sendFeedback = async (borough: string, feedback: string) => {
    // TODO: send feedback to backend
  }

  if (!boroughs) return null

  const top = recommendations[0]
  const rest = recommendations.slice(1)
  const topBorough = boroughs.find(
    (b) => b.name.toLowerCase() === top?.borough.toLowerCase(),
  )

  const rent = getRentIndicator(top?.norm_rent)
  const crime = getCrimeIndicator(top?.norm_crime)
  const youth = getYouthIndicator(top?.norm_youth)

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <h1 className="mb-10 text-3xl font-bold">Your Top Borough Matches</h1>

      {top && topBorough && (
        <>
          <Link
            href={`/borough/${topBorough.slug}`}
            className="mb-4 block items-center gap-6 rounded-3xl border border-zinc-700 bg-zinc-900 p-6 shadow-xl transition hover:scale-[1.01] md:flex"
          >
            <div className="relative mb-4 aspect-video w-full overflow-hidden rounded-2xl md:mb-0 md:w-1/2">
              <img
                src={topBorough.image}
                alt={topBorough.name}
                className="h-full w-full object-cover"
              />
            </div>
            <div className="text-white md:w-1/2">
              <p className="mb-1 text-sm font-semibold uppercase tracking-wide text-green-400">
                Best match!
              </p>
              <h2 className="mb-3 text-3xl font-bold">
                {toTitleCase(topBorough.name)}
              </h2>
              <p className="mb-4 text-gray-300">
                This borough matches{' '}
                <span className="font-semibold text-green-400">
                  {(top.score * 100).toFixed(1)}%
                </span>{' '}
                of your preferences.
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-4 text-sm text-gray-400">
                {rent && (
                  <div className="flex items-center gap-1">
                    <span className={`flex items-center ${rent.color}`}>
                      {rent.icon}
                    </span>
                    <span>{rent.label}</span>
                  </div>
                )}
                {crime && (
                  <div className="flex items-center gap-1">
                    <span className={`flex items-center ${crime.color}`}>
                      {crime.icon}
                    </span>
                    <span>{crime.label}</span>
                  </div>
                )}
                {youth && (
                  <div className="flex items-center gap-1">
                    <span className="flex items-center">{youth.icon}</span>
                    <span>{youth.label}</span>
                  </div>
                )}
              </div>
            </div>
          </Link>

          <div className="mb-14 mr-2 flex flex-col items-end text-sm text-gray-500">
            <div className="flex items-center gap-3">
              <button
                onClick={() => sendFeedback(topBorough.name, 'like')}
                className="flex items-center gap-1 text-gray-400 transition hover:text-green-400"
              >
                <ThumbsUp className="h-4 w-4" strokeWidth={1.5} />
                Yes
              </button>
              <button
                onClick={() => sendFeedback(topBorough.name, 'dislike')}
                className="flex items-center gap-1 text-gray-400 transition hover:text-red-400"
              >
                <ThumbsDown className="h-4 w-4" strokeWidth={1.5} />
                No
              </button>
            </div>
          </div>
        </>
      )}

      <h3 className="mb-4 text-xl font-semibold text-white">
        Other suggestions
      </h3>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
        {rest.map((rec) => {
          const details = boroughs.find(
            (b) => b.name.toLowerCase() === rec.borough.toLowerCase(),
          )
          if (!details) return null

          return (
            <BoroughCard
              key={details.slug}
              slug={details.slug}
              name={details.name}
              image={details.image}
              scoreLabel={`Score: ${(rec.score * 100).toFixed(0)}%`}
              norm_rent={details.norm_rent}
              norm_crime={details.norm_crime}
              norm_youth={details.norm_youth}
            />
          )
        })}
      </div>
    </div>
  )
}
