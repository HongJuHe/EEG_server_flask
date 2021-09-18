  var mySwiper = new Swiper('.swiper-container', {
    // Swipe Button for change Div
    navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
    },
        
    // Blue Point for Current Div
    pagination: {
        el: '.swiper-pagination',
        type: 'bullets',
    },
    
});