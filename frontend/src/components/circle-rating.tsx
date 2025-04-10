type CircleRatingProps = {
  score: number
  editable?: boolean
  onChange?: (score: number) => void
}

export default function CircleRating({
  score,
  editable = false,
  onChange,
}: CircleRatingProps) {
  return (
    <div className="flex gap-1">
      {Array.from({ length: 5 }, (_, i) => {
        const filled = i < score

        const getColor = () => {
          if (!filled) return 'bg-zinc-600'
          if (score >= 4) return 'bg-green-400'
          if (score >= 3) return 'bg-yellow-400'
          if (score >= 2) return 'bg-orange-400'
          return 'bg-red-400'
        }

        return (
          <button
            key={i}
            onClick={() => editable && onChange?.(i + 1)}
            className={`h-4 w-4 rounded-full transition ${getColor()}`}
          />
        )
      })}
    </div>
  )
}
