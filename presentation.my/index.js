// import Vue from 'vue';

var app = new Vue({
    el: '#app',
    data: {
        w_height: 0,
        w_width: 0,
        transform_tleft: 0,
        transform_ttop: 0,
        transform_scale: 1,
    },
    methods: {
    },
    mounted() {
        console.log("fuck");
        this.w_height = window.innerHeight;
        this.w_width = window.innerWidth;
        box = this.$el.getBoundingClientRect()
        ratio = window.innerWidth / box.width;
        ratio = Math.min(window.innerHeight / box.height, ratio);
        this.transform_scale *= ratio;
        this.transform_tleft -= box.left;
        this.transform_ttop += - box.top * ratio

        transform = 'translate(' + this.transform_tleft +
            'px,' + this.transform_ttop + 'px) ' + ' scale(' + this.transform_scale + ') ';
        console.log(transform);
        origin = ' top left';
        document.body.style.transformOrigin = origin;
        document.body.style.OTransformOrigin = origin;
        document.body.style.msTransformOrigin = origin;
        document.body.style.MozTransformOrigin = origin;
        document.body.style.WebkitTransformOrigin = origin;

        document.body.style.transform = transform;
        document.body.style.OTransform = transform;
        document.body.style.msTransform = transform;
        document.body.style.MozTransform = transform;
        document.body.style.WebkitTransform = transform;

    }
})
