'use client'

import {
  getCrimeIndicator,
  getRentIndicator,
  getYouthIndicator,
  toTitleCase,
} from '@/lib/utils'
import Image from 'next/image'
import Link from 'next/link'

type Props = {
  slug: string
  name: string
  image: string
  norm_rent?: number
  norm_crime?: number
  norm_youth?: number
  scoreLabel?: string
}

export default function BoroughCard({
  slug,
  name,
  image,
  scoreLabel,
  norm_rent,
  norm_crime,
  norm_youth,
}: Props) {
  const rent = getRentIndicator(norm_rent)
  const crime = getCrimeIndicator(norm_crime)
  const youthInfo = getYouthIndicator(norm_youth)

  return (
    <Link
      href={`/borough/${slug}`}
      className="relative overflow-hidden rounded-2xl bg-zinc-900 shadow-lg transition hover:scale-105"
    >
      {scoreLabel && (
        <span className="absolute left-2 top-2 z-10 rounded-full bg-white px-2 py-0.5 text-xs font-semibold text-black shadow">
          {scoreLabel}
        </span>
      )}
      <div className="relative h-48 w-full">
        <Image
          src={image ?? '/images/london-background.png'}
          alt={name}
          fill
          className="object-cover"
        />
      </div>
      <div className="p-4 text-left">
        <h3 className="mb-1 text-xl font-semibold text-white">
          {toTitleCase(name)}
        </h3>
        <div className="mt-2 flex flex-wrap items-center gap-4 text-sm text-gray-400">
          {rent && (
            <div className="flex gap-1">
              <span className={`flex items-center ${rent.color}`}>
                {rent.icon}
              </span>
              <span> {rent.label}</span>
            </div>
          )}
          {crime && (
            <div className="flex gap-1">
              <span className={`flex items-center ${crime.color}`}>
                {crime.icon}
              </span>
              <span> {crime.label}</span>
            </div>
          )}
          {youthInfo && (
            <div className="flex gap-1">
              <span className="flex items-center">{youthInfo.icon}</span>
              <span> {youthInfo.label}</span>
            </div>
          )}
        </div>
      </div>
    </Link>
  )
}
