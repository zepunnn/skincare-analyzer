// main.js — shared logic across all pages (Home, Analyze, About)

document.addEventListener("DOMContentLoaded", () => {
  // Highlight the active nav link based on current page filename
  const currentPage = window.location.pathname.split("/").pop() || "index.html";
  const navLinks = document.querySelectorAll(".topbar-nav a");

  navLinks.forEach((link) => {
    const linkPage = link.getAttribute("href");
    if (linkPage === currentPage) {
      link.classList.add("active");
    }
  });
});
