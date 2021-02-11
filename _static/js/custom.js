jQuery(function ($) {
  $("[data-toggle='popover']").popover({trigger: "hover click", delay: {"show": 0, "hide": 500}}).click(function (event) {
    event.stopPropagation();

  }).on('inserted.bs.popover', function () {
    $(".popover").click(function (event) {
      event.stopPropagation();
    })
  })

  $(document).click(function () {
    $("[data-toggle='popover']").popover('hide')
  })
})