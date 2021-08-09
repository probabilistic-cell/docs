// jQuery(function ($) {
//   $("[data-toggle='popover']").popover({trigger: "hover click", delay: {"show": 0, "hide": 500}}).click(function (event) {
//     event.stopPropagation();

//   }).on('inserted.bs.popover', function () {
//     $(".popover").click(function (event) {
//       event.stopPropagation();
//     })
//   })

//   $(document).click(function () {
//     $("[data-toggle='popover']").popover('hide')
//   })
// })




$(function(){
  $("[data-toggle=popover]").popover({
      html : true,
      content: function() {
        var content = $(this).attr("data-popover-content");
        return $(content).children(".popover-body").html();
      },
      title: function() {
        var title = $(this).attr("data-popover-content");
        return $(title).children(".popover-heading").html();
      }
  });

  $(document).click(function () {
    $("[data-toggle='popover']").popover('hide')
  })
});