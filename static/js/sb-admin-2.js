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
  var files, formData;
  const handleImageUpload = event => {
    files = event.target.files
    formData = new FormData()
    formData.append('img', files[0])

    $('.content-01').show();
    $('.spinner-border').show();

    fetch('/save_local', {
      method: 'POST',
      body: formData
    }).then(data => {

      fetch('/upload', {
        method: 'POST',
        body: formData
      }).then(data => {

        fetch('/recog', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {

          // console.log(data)

          var table_available = data.available;
          var table_missing = data.not_available;
          var chart_boxes = data._all_boxes;

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

          $(".data-ink-comment").html(data.data_ink_ratio_comment)
          $(".bck-comment").html(data.background_score_comment)
          // $(".x-axis-comment").html(data.x_spread_ratio_comment)
          // $(".y-axis-comment").html(data.y_spread_ratio_comment)

          $(".chart-elem-p").html(String(data.chart_elem_score).concat("%"))

          if (data.data_ink_ratio_score < 33) { $(".data-ink-bar").addClass("bg-danger") }
          else if (data.data_ink_ratio_score >= 33 && data.data_ink_ratio_score < 66) { $(".data-ink-bar").addClass("bg-warning") }
          else { $(".data-ink-bar").addClass("bg-success") }

          if (data.spacing_score < 33) { $(".spacing-bar").addClass("bg-danger") }
          else if (data.spacing_score >= 33 && data.spacing_score < 66) { $(".spacing-bar").addClass("bg-warning") }
          else { $(".spacing-bar").addClass("bg-success") }

          if (data.chart_elem_score < 33) { $(".chart-elem-bar").addClass("bg-danger") }
          else if (data.chart_elem_score >= 33 && data.chart_elem_score < 66) { $(".chart-elem-bar").addClass("bg-warning") }
          else { $(".chart-elem-bar").addClass("bg-success") }

          if (data.background_score < 33) { $(".background-bar").addClass("bg-danger") }
          else if (data.background_score >= 33 && data.background_score < 66) { $(".background-bar").addClass("bg-warning") }
          else { $(".background-bar").addClass("bg-success") }

          if (data.overall_score < 33) { $(".overall-bar").addClass("bg-danger") }
          else if (data.overall_score >= 33 && data.overall_score < 66) { $(".overall-bar").addClass("bg-warning") }
          else { $(".overall-bar").addClass("bg-success") }

          $(".img-fluid-custom").attr("src", "static/images/".concat(files[0].name));

          //boxes

          chart_boxes.forEach(elem => {

            $(".boxes-all").append("<div style='border: 3px solid #d9534f; position: absolute; top:" + String((parseInt(elem[1]) +70)) + "px; left: " + String(parseInt(elem[0] + 70)) + "px; width: " + elem[2] + "px; height: " + elem[3] +"px;'>" );

            
          });
        
          $('.content-01').hide();
          $('.content-02').show();

          $('.spinner-border').hide();

        }).catch(error => {
          console.error(error)
        })


      }).catch(error => {
        console.error(error)
      })


    }).catch(error => {
      console.error(error)
    })
  }

  document.querySelector('#file').addEventListener('change', event => {
    handleImageUpload(event)
  })

  $('.content-02').hide();
  $('.spinner-border').hide();

})(jQuery); // End of use strict
