// Removing 'made with chainlit' watermark

document.addEventListener('DOMContentLoaded', function() {
    // Function to remove the target element
    function removeTargetElement() {
      // Select the target element by its class
      const targetElement = document.querySelector('.MuiStack-root.watermark');
      console.log(targetElement);
      
      if (targetElement) {
        // Removing the target element from the DOM
        targetElement.remove();
  
        // Console verification
        console.log('Custom script executed successfully!');
      } else {
        // Console error if the target element is not found
        console.error('Target element not found!');
      }
    }
  
    // Create a MutationObserver to watch for changes in the DOM
    const observer = new MutationObserver(function(mutationsList, observer) {
      for (const mutation of mutationsList) {
        if (mutation.type === 'childList') {
          // Try to find the target element whenever child nodes are added
          const targetElement = document.querySelector('.MuiStack-root.watermark');
          if (targetElement) {
            // If the target element is found, remove it and stop observing
            removeTargetElement();
            observer.disconnect();
          }
        }
      }
    });
  
    // Start observing the entire document for child node changes
    observer.observe(document.body, { childList: true, subtree: true });
  
    // Call the function initially in case the element is already present
    removeTargetElement();
});
