import Link from "next/link";

export default function Page() {
  return (
    <main className="flex flex-col gap-8 justify-center ">
      <div
        className={` text-5xl font-medium capitalize tracking-wide w-full italic `}
      >
        <span className=" drop-shadow-md font-extrabold text-blue-500">
          Stock
        </span>
        <span className=" ">Vision</span>
        <div className=" text-sm mt-5">
          ~ Predictive Analytics and <br />
          Portfolio Management
        </div>
      </div>

      <ol className="list-inside list-decimal text-sm text-center italic sm:text-left ">
        <li className="mb-2">Predict Stock Price</li>
        <li>Get Portfolio Suggestion</li>
      </ol>

      <div className="grid grid-cols-2 gap-2">
        <Link
          className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
          href="/stock-prediction"
        >
          Stock Prediction
        </Link>
        <Link
          className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44"
          href="/portfolio-suggestion"
        >
          Portfolio Suggestion
        </Link>
      </div>
    </main>
  );
}
