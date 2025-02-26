"use client";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const [toggleMenu, setToggleMenu] = useState(true);
  const pathname = usePathname();

  return (
    <nav className={` px-8 py-4 `}>
      <div className=" max-w-screen-2xl flex flex-wrap items-center justify-between mx-auto ">
        <Link
          href="/"
          className="flex items-center space-x-3 rtl:space-x-reverse"
        >
          <div className="flex text-2xl">
            <span className=" drop-shadow-md font-extrabold text-blue-500">
              Stock
            </span>
            <span className=" ">Vision</span>
          </div>
        </Link>

        <div className="flex md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse md:hidden ">
          <button
            className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm rounded-lg md:hidden "
            onClick={() => setToggleMenu(!toggleMenu)}
          >
            <span className="sr-only">Open main menu</span>
            <svg
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 17 14"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M1 1h15M1 7h15M1 13h15"
              />
            </svg>
          </button>
        </div>

        <div
          className={`items-center justify-between w-full md:flex md:w-auto md:order-1 ${
            toggleMenu ? "hidden" : ""
          }`}
        >
          <ul className="flex gap-1 flex-col font-medium p-4 md:p-0 mt-4 rounded-lg bg-secondary-background  md:bg-transparent  md:space-x-8 rtl:space-x-reverse md:flex-row md:mt-0">
            <li>
              <Link
                href="/stock-prediction"
                className={`py-2 px-3 md:p-0 rounded md:bg-transparent  flex gap-2  items-center  ${
                  pathname === "/stock-prediction" && " text-blue-500  "
                }`}
              >
                Stock Prediction
              </Link>
            </li>
            <li>
              <Link
                href="/portfolio-suggestion"
                className={`py-2 px-3 md:p-0 rounded md:bg-transparent  flex gap-2  items-center ${
                  pathname === "/portfolio-suggestion" && " text-blue-500  "
                }`}
              >
                Portfolio Suggestion
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
