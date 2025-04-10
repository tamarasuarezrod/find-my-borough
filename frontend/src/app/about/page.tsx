export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-12 text-white">
      <h1 className="mb-4 text-4xl font-bold">About Find My Borough</h1>
      <p className="mb-10 text-lg text-gray-300">
        Want to live in London but not sure where to start? Or maybe you're
        already here and curious about discovering new boroughs? You're not
        alone. London has a total of <strong>33 boroughs</strong>, each with its
        own personality, pace and charm, from the buzzing streets of Camden to
        the leafy calm of Richmond.
      </p>
      <p className="mb-10 text-lg text-gray-300">
        <span className="font-medium text-white">Find My Borough</span> is here
        to bring all of them closer to you. Whether you're moving, exploring or
        just dreaming, this platform helps you understand what each area has to
        offer, and even recommends matches based on your lifestyle.
      </p>

      <section className="mb-10">
        <h2 className="mb-2 text-2xl font-semibold">How it works</h2>
        <p className="text-gray-400">
          Just answer a few questions about your budget, lifestyle, priorities
          and plans. Our system uses real data and a machine learning model to
          recommend the boroughs that best match your preferences, even if
          you're not familiar with the city.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="mb-2 text-2xl font-semibold">✨ Powered by AI</h2>
        <p className="text-gray-400">
          Under the hood, Find My Borough uses a custom-built AI model that
          learns from your answers and applies them to borough data. It's like a
          local friend who knows the whole city, and just wants to help you find
          your place in it.
        </p>
      </section>

      <section className="mt-20 text-center">
        <p className="text-gray-400">
          Got questions, feedback, or just want to say hi? Send an email to{' '}
          <a
            href="mailto:contact@findmyborough.uk"
            className="font-medium text-white underline hover:text-gray-200"
          >
            contact@findmyborough.uk
          </a>
        </p>
      </section>
    </div>
  )
}
