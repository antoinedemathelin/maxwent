// Adjust scrolling to account for the fixed header
document.querySelectorAll('.toc-container a').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    
    // Get the target element
    const targetId = this.getAttribute('href').substring(1); // Remove the '#' part
    const targetElement = document.getElementById(targetId);
    
    // Scroll to the target element with an offset to account for the fixed header
    window.scrollTo({
      top: targetElement.offsetTop - 70, // Adjust '50' to match your header height
      behavior: 'smooth' // Smooth scrolling
    });
  });
});



document.addEventListener("scroll", () => {
    const sections = document.querySelectorAll("main h1, main h2, main h3, main h4, main h5, main h6");
    const tocLinks = document.querySelectorAll(".toc-container a");

    let currentSectionId = null;

    sections.forEach((section) => {
        const rect = section.getBoundingClientRect();

        // Check if the section's top is above the middle of the viewport
        if (rect.top <= window.innerHeight / 2) {
            currentSectionId = section.id;
        }
    });

    // Update the toc-container links to reflect the current section
    tocLinks.forEach((link) => {
        const href = link.getAttribute("href").substring(1); // Remove '#' from href
        if (href === currentSectionId) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
});




