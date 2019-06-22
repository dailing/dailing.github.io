// import Vue from 'vue';


Vue.component('slide', {
    data: function () {
        return {
            title: 'Title',
            slide_index:-1,
        }
    },
    props: ['title'],
    methods: {
    },
    mounted(){
        // console.log(this);
        me_index = this.$parent.register_slide(this);
        this.slide_index = me_index;
    },
    template: 
`
<div class="border border-warning slide " 
        v-bind:class="{ active: $parent.current_slide == slide_index}">
    <slot name='header'>
        <div class="slide_head">
            <h1>{{ title }}</h1>
        </div>
    </slot>
    <div class="slide_content">
        <slot></slot>
    </div>
    <div class="page_number">
        <a v-on:click="$parent.prev()" > < </a>
        {{ slide_index }}
        <a v-on:click="$parent.next()" > > </a>
    </div>
</div>
`,
})


var app = new Vue({
    el: '#app',
    data: {
        message: 'Hello Vue!',
        slide_count : 0,
        slides : [],
        current_slide : null,
        current_url : null,
        w_height:0,
        w_width:0,
        transform_tleft:0,
        transform_ttop:0,
        transform_scale:1
    },
    methods:{
        register_slide(child){
            this.slide_count += 1;
            this.slides.push(child);
            return this.slide_count - 1;
        },
        get_url_for_page(i){
            return this.current_url + "#" + i;
        },
        goto(page){
            // return;
            if (this.transform_tleft === null){
                box = this.slides[0].$el.getBoundingClientRect();
                this.transform_tleft = box.left;
                this.transform_ttop = box.top;
            }
            console.log('goto ' + page);
            prev_page = this.current_slide;
            page = Math.min(page, this.slide_count-1);
            page = Math.max(page, 0);
            this.current_slide = page;
            window.history.replaceState(null, null, this.get_url_for_page(this.current_slide));
            if(prev_page != page){
                // animation here
                box = this.slides[page].$el.getBoundingClientRect()
                ratio = window.innerWidth / box.width;
                ratio = Math.min(window.innerHeight / box.height, ratio);
                this.transform_scale *= ratio;
                this.transform_tleft -= box.left;
                this.transform_ttop +=  - box.top * ratio

                transform = 'translate('+ this.transform_tleft +
                    'px,' + this.transform_ttop + 'px) ' + ' scale(' +  this.transform_scale + ') ';
                console.log(transform);
                // origin =  window.innerHeight / 2 - box.height * ratio / 2 + 'px left';
                origin =  ' top left';
                // console.log(this.slides[page].$el.style)
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
        },
        prev(){
            this.goto(this.current_slide - 1);
        },
        next(title=null){
            this.goto(this.current_slide + 1);
        },
    },
    mounted () {
        url_and_query = window.location.href.split('#');
        this.current_url = url_and_query[0];
        if(url_and_query.length > 1){
            page = parseInt(url_and_query[1])
            if(!isNaN(page)){
                console.log('goto page ' + page);
                this.goto(page);
            }else{
                console.log('error, parsing:' + page)
            }
        }else{
            this.goto(0);
        }
        this.goto(this.current_slide);
        this.w_height = window.innerHeight;
        this.w_width = window.innerWidth;
        console.log('get current url ' + this.current_url);

    }
})
