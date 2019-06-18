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
    mounted : function(){
        console.log(this);
        me_index = this.$parent.register_slide();
        this.slide_index = me_index;
    },
    template: 
`
<div class="border border-warning slide" v-if="$parent.current_slide==slide_index">
    <slot name='header'>
        <div class="slide_head">
            <h1>{{ title }}</h1>
        </div>
    </slot>
    <div class="slide_content" style="height: 100%">
        <slot></slot>
    </div>
    <div class="page_number">
        <a v-on:click="$parent.prev()"> <= </a>
        {{ slide_index }}
        <a v-on:click="$parent.next()"> => </a>
    </div>
</div>
`,
})


var app = new Vue({
    el: '#app',
    data: {
        message: 'Hello Vue!',
        slide_count : 0,
        current_slide : null,
        current_url : null,
    },
    methods:{
        register_slide(){
            this.slide_count += 1;
            return this.slide_count - 1;
        },
        get_url_for_page(i){
            return this.current_url + "#" + i;
        },
        goto(page){
            page = Math.min(page, this.slide_count-1);
            page = Math.max(page, 0);
            this.current_slide = page;
            window.history.replaceState(null, null, this.get_url_for_page(this.current_slide));
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
        console.log(this);
        console.log('get current url ' + this.current_url);
    }
})
