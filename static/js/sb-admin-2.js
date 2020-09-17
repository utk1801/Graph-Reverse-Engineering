(function ($) {
  "use strict"; // Start of use strict

  // Toggle the side navigation
  $("#sidebarToggle, #sidebarToggleTop").on('click', function (e) {
    $("body").toggleClass("sidebar-toggled");
    $(".sidebar").toggleClass("toggled");
    if ($(".sidebar").hasClass("toggled")) {
      $('.sidebar .collapse').collapse('hide');
    };
  });

  // Close any open menu accordions when window is resized below 768px
  $(window).resize(function () {
    if ($(window).width() < 768) {
      $('.sidebar .collapse').collapse('hide');
    };
  });

  // Prevent the content wrapper from scrolling when the fixed side navigation hovered over
  $('body.fixed-nav .sidebar').on('mousewheel DOMMouseScroll wheel', function (e) {
    if ($(window).width() > 768) {
      var e0 = e.originalEvent,
        delta = e0.wheelDelta || -e0.detail;
      this.scrollTop += (delta < 0 ? 1 : -1) * 30;
      e.preventDefault();
    }
  });

  // Scroll to top button appear
  $(document).on('scroll', function () {
    var scrollDistance = $(this).scrollTop();
    if (scrollDistance > 100) {
      $('.scroll-to-top').fadeIn();
    } else {
      $('.scroll-to-top').fadeOut();
    }
  });

  // Smooth scrolling using jQuery easing
  $(document).on('click', 'a.scroll-to-top', function (e) {
    var $anchor = $(this);
    $('html, body').stop().animate({
      scrollTop: ($($anchor.attr('href')).offset().top)
    }, 1000, 'easeInOutExpo');
    e.preventDefault();
  });

  // Custom

  const handleImageUpload = event => {
    const files = event.target.files
    const formData = new FormData()
    formData.append('myFile', files[0])

    $('.content-01').show();

    fetch('/api1', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json()) {

      }

    co

    fetch('/chart_data', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {

        var table_available = data.available
        var table_missing = data.not_available

        table_available.forEach(elem => {
          $('.'.concat(elem, '-available')).show();
          $('.'.concat(elem, '-missing')).hide();
        });

        table_missing.forEach(elem => {
          $('.'.concat(elem, '-available')).hide();
          $('.'.concat(elem, '-missing')).show();
        });

        $(".data-ink-bar").css("width", String(data.data_ink_ratio_score).concat("%"));
        $(".spacing-bar").css("width", String(data.spacing_score).concat("%"));
        $(".chart-elem-bar").css("width", String(data.chart_elem_score).concat("%"));
        $(".background-bar").css("width", String(data.background_score).concat("%"));
        $(".overall-bar").css("width", String(data.overall_score).concat("%"));

        $(".data-ink-p").html(String(data.data_ink_ratio_score).concat("%"))
        $(".spacing-p").html(String(data.spacing_score).concat("%"))
        $(".chart-elem-p").html(String(data.chart_elem_score).concat("%"))
        $(".background-p").html(String(data.background_score).concat("%"))
        $(".overall-p").html(String(data.overall_score).concat("%"))

        if (data.data_ink_ratio_score < 33) {$(".data-ink-bar").addClass("bg-danger")} 
        else if (data.data_ink_ratio_score >= 33 && data.data_ink_ratio_score < 66) {$(".data-ink-bar").addClass("bg-warning")}
        else {$(".data-ink-bar").addClass("bg-success")}

        if (data.spacing_score < 33) {$(".spacing-bar").addClass("bg-danger")} 
        else if (data.spacing_score >= 33 && data.spacing_score < 66) {$(".spacing-bar").addClass("bg-warning")}
        else {$(".spacing-bar").addClass("bg-success")}

        if (data.chart_elem_score < 33) {$(".chart-elem-bar").addClass("bg-danger")} 
        else if (data.chart_elem_score >= 33 && data.chart_elem_score < 66) {$(".chart-elem-bar").addClass("bg-warning")}
        else {$(".chart-elem-bar").addClass("bg-success")}

        if (data.background_score < 33) {$(".background-bar").addClass("bg-danger")} 
        else if (data.background_score >= 33 && data.background_score < 66) {$(".background-bar").addClass("bg-warning")}
        else {$(".background-bar").addClass("bg-success")}

        if (data.overall_score < 33) {$(".overall-bar").addClass("bg-danger")} 
        else if (data.overall_score >= 33 && data.overall_score < 66) {$(".overall-bar").addClass("bg-warning")}
        else {$(".overall-bar").addClass("bg-success")}

        $(".img-fluid-custom").attr("src", "static/images/".concat(files[0].name));

        $('.content-01').hide();
        $('.content-02').show();

      })
      .catch(error => {
        console.error(error)
      })
    // fetch('/saveImage', {
    //   method: 'POST',
    //   body: formData
    // })
    // .then(response => response.json())
    // .then(data => {
    //   console.log(data.path)
    // })
    // .catch(error => {
    //   console.error(error)
    // })
  }

  document.querySelector('#file').addEventListener('change', event => {
    handleImageUpload(event)
  })

  $('.content-02').hide();

  //   document.getElementById("file").onchange = function() {
  //     document.getElementById("form").submit();
  // }

})(jQuery); // End of use strict
