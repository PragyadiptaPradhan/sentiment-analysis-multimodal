"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { FiMoon, FiSun } from "react-icons/fi";

export function DarkModeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch
  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return <div className="h-8 w-16 rounded-full bg-gray-200 dark:bg-gray-700" />;
  }

  const isDark = theme === "dark";

  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label="Toggle dark mode"
      className="relative flex h-8 w-16 items-center rounded-full border border-gray-300 bg-gray-100 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-1 dark:border-gray-600 dark:bg-gray-700"
    >
      {/* Track fill */}
      <span
        className={`absolute inset-0 rounded-full transition-colors duration-300 ${isDark ? "bg-gray-700" : "bg-gray-200"}`}
      />
      {/* Icons */}
      <FiSun className="relative z-10 ml-1.5 h-4 w-4 text-yellow-500" />
      <FiMoon className="relative z-10 ml-auto mr-1.5 h-4 w-4 text-gray-400 dark:text-blue-300" />
      {/* Thumb */}
      <span
        className={`absolute h-6 w-6 rounded-full bg-white shadow-md transition-transform duration-300 ${isDark ? "translate-x-8" : "translate-x-1"}`}
      />
    </button>
  );
}
