export default function PrivacyPolicy() {
  return (
    <div className="mx-auto max-w-4xl px-6 py-12 text-white">
      <div className="mb-8 flex justify-center"></div>
      <h1 className="mb-4 text-4xl font-bold">Privacy Policy</h1>
      <p className="mb-6 text-gray-300">
        FindMyBorough uses third-party providers (Google and Facebook) to
        authenticate users. When you log in, we receive your name and email
        address to create your account.
      </p>
      <p className="mb-6 text-gray-300">
        Some features of the app, such as "Find My Match" and borough ratings,
        require users to be logged in. This allows us to associate your inputs
        with your account to improve the recommendation experience.
      </p>
      <p className="text-gray-300">
        If you have any questions or would like your account deleted, please
        contact us at{' '}
        <a href="mailto:findmyborough@gmail.com" className="underline">
          contact@findmyborough.uk
        </a>
      </p>
    </div>
  )
}
