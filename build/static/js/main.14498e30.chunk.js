(window["webpackJsonpmy-app"]=window["webpackJsonpmy-app"]||[]).push([[0],{196:function(e,t,a){e.exports=a(324)},32:function(e,t,a){e.exports=a.p+"static/media/default.0c8671a5.png"},324:function(e,t,a){"use strict";a.r(t);var n=a(23),i=a(24),r=a(28),s=a(25),l=a(27),o=a(0),c=a.n(o),m=a(6),p=a.n(m),u=a(180),h=a(384),d=a(179),_=a.n(d),g=a(178),f=a.n(g),v=a(13),y=a(363),b=a(333),E=a(373),k=a(328),w=a(327),x=a(332),O=a(371),C=a(365),S=a(164),j=a.n(S);function F(e){var t=Object(y.a)((function(t){return{container:{display:"flex",flexWrap:"wrap"},textField:{display:"flex",width:e.width,marginTop:e.marginTop,margin:"auto"},button:{display:"flex",width:e.width,marginTop:e.marginTop,margin:"auto"},FormHelperText:{display:"flex",width:e.width,margin:"auto",marginTop:7,color:"red "},input:{display:"none"},rightIcon:{marginLeft:t.spacing(1)},iconSmall:{fontSize:20}}}))(),n=c.a.useState(""),i=Object(v.a)(n,2),r=i[0],s=i[1],l=c.a.useState(""),o=Object(v.a)(l,2),m=o[0],p=o[1];return e.values.name===e.types[0]?c.a.createElement("div",null,c.a.createElement(C.a,{label:"Number of frames",id:"margin-normal",defaultValue:"",className:t.textField,onChange:function(e){p(e.target.value)},helperText:"",margin:"normal"}),c.a.createElement(C.a,{label:"ref to "+e.values.name,id:"margin-normal",defaultValue:"",className:t.textField,onChange:function(e){s(e.target.value)},helperText:"",margin:"normal"}),c.a.createElement(O.a,{variant:"outlined",color:"primary",className:t.button,onClick:function(t){t.preventDefault();var n=e.default_check();if(!0===n){var i=isNaN(parseInt(m));if(!0===i)return i=!1,void e.seterror_mesage(e.error_mesage_strings[2]);i=!0;var s=!1,l=!1;if(void 0!==j.a.parse(r)&&(s=!0),!1===s){var o=r.indexOf(".link");0===r.indexOf("http://")&&o===r.length-5&&(l=!0)}if(!1===s&&!1===l)0===r.indexOf("rtsp://")&&!0;if(!1===s&&!1===l&&!1===l){var c=r.indexOf(".mjpg");if(0!==r.indexOf("http://")||c!==r.length-5)return void e.seterror_mesage(e.error_mesage_strings[3]);!0}a(69)({method:"post",url:"",data:{ref:r,nn:e.nn,num_of_frames:m}}).then((function(t){e.set_im_src()})).catch((function(e){return console.warn(e)})),e.set_page_to_1()}else e.seterror_mesage(n)}},"Go"),c.a.createElement(k.a,{className:t.FormHelperText},e.error_mesage)):null}var I=a(166),N=a.n(I),T=a(372);function P(e){var t=Object(y.a)((function(t){return{div:{display:"flex",flexWrap:"wrap"},textField:{display:"flex",width:e.width,marginTop:e.marginTop,margin:"auto"},button:{display:"flex",width:e.width,marginTop:e.marginTop,margin:"auto"},FormHelperText:{display:"flex",width:e.width,margin:"auto",marginTop:7,color:"red "},input:{display:"none"},rightIcon:{marginLeft:t.spacing(1)},iconSmall:{fontSize:20}}}))(),n=c.a.useState(""),i=Object(v.a)(n,2),r=i[0],s=i[1],l=c.a.useState(""),o=Object(v.a)(l,2),m=(o[0],o[1]),p=a(221),u=c.a.useState(p()),h=Object(v.a)(u,2),d=(h[0],h[1],"");return e.values.name===e.types[1]?c.a.createElement("div",{key:p()},c.a.createElement(T.a,{key:p()},c.a.createElement("input",{accept:"video/*",className:t.input,id:"contained-button-file",multiple:!0,type:"file",name:"inputFile",ref:function(e){return d=e},onChange:function(e){e.preventDefault(),s(d.files[0]),m(d.files[0].name)}})),c.a.createElement("label",{htmlFor:"contained-button-file"},c.a.createElement(O.a,{variant:"contained",component:"span",className:t.button},"Upload",c.a.createElement(N.a,{className:t.rightIcon})),c.a.createElement(O.a,{type:"submit",variant:"outlined",color:"primary",className:t.button,onClick:function(t){t.preventDefault();var n=e.default_check();if(!0===n)if(""!==r){var i=r,s=new FormData;s.append("video",i),function(t){var n=a(69);n.post("/",t).then((function(t){return e.set_im_src()}),n({method:"post",url:"",data:{nn:e.nn}})).catch((function(e){return console.warn(e)}))}(s),e.set_page_to_1()}else e.seterror_mesage(e.error_mesage_strings[0]);else e.seterror_mesage(n)}},"Go"),c.a.createElement(k.a,{className:t.FormHelperText},e.error_mesage))):null}var M=a(61),W=a(10);a(274),a(68);function H(e){console.log("onOpenChange",e)}var D=[c.a.createElement(W.b,{title:c.a.createElement("span",{className:"submenu-title-wrapper"},"Choose type of detector"),key:"0",popupOffset:[10,15],style:{width:"100%",backgroundColor:"white"}},c.a.createElement(W.a,{key:"1"},"enet-coco"),c.a.createElement(W.b,{key:"2-2",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"yolov3")},c.a.createElement(W.a,{key:"2"},"coco"),c.a.createElement(W.a,{key:"3"},"drone"),c.a.createElement(W.a,{key:"4"},"openimages")),c.a.createElement(W.b,{key:"3-3",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"yolov3-tiny")},c.a.createElement(W.a,{key:"5"},"coco"),c.a.createElement(W.a,{key:"6"},"drone")),c.a.createElement(W.b,{key:"4-4",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"yolov3-spp")},c.a.createElement(W.a,{key:"7"},"coco"),c.a.createElement(W.b,{key:"4-4-4",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"drone")},c.a.createElement(W.a,{key:"8"},"spp1"),c.a.createElement(W.a,{key:"9"},"spp3"))),c.a.createElement(W.b,{key:"5-5",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"yolov3-spp-slim")},c.a.createElement(W.b,{key:"5-5-5",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"drone")},c.a.createElement(W.a,{key:"10"},"prune_0.5"),c.a.createElement(W.a,{key:"11"},"prune_0.5_0.5_0.7"),c.a.createElement(W.a,{key:"12"},"prune_0.9"),c.a.createElement(W.a,{key:"13"},"prune_0.95"))),c.a.createElement(W.b,{key:"6-6",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"ttfnet")},c.a.createElement(W.b,{key:"6-6-6",title:c.a.createElement("span",{className:"submenu-title-wrapper"},"coco")},c.a.createElement(W.a,{key:"14"},"ttfnet_d53_1x"),c.a.createElement(W.a,{key:"15"},"ttfnet_d53_2x"),c.a.createElement(W.a,{key:"16"},"ttfnet_r18_1x"),c.a.createElement(W.a,{key:"17"},"ttfnet_r18_2x"),c.a.createElement(W.a,{key:"18"},"ttfnet_r34_2x"))))],z=c.a.createElement("span",null,"Add More Items"),B=function(e){function t(e){var a;return Object(n.a)(this,t),(a=Object(r.a)(this,Object(s.a)(t).call(this,e))).state={children:D,overflowedIndicator:void 0},a.toggleOverflowedIndicator=function(){a.setState({overflowedIndicator:void 0===a.state.overflowedIndicator?z:void 0})},a.handleClick=a.handleClick.bind(Object(M.a)(a)),a}return Object(l.a)(t,e),Object(i.a)(t,[{key:"handleClick",value:function(e){console.log("clicked ".concat(e.key)),console.log(e),this.props.set_nn(e.key)}},{key:"render",value:function(){var e=this.props.triggerSubMenuAction,t=this.state,a=t.children,n=t.overflowedIndicator;return c.a.createElement("div",null,this.props.updateChildrenAndOverflowedIndicator&&c.a.createElement("div",null,c.a.createElement("button",{onClick:this.toggleChildren},"toggle children"),c.a.createElement("button",{onClick:this.toggleOverflowedIndicator},"toggle overflowedIndicator")),c.a.createElement(W.c,{onClick:this.handleClick,triggerSubMenuAction:e,onOpenChange:H,selectedKeys:["0"],mode:this.props.mode,openAnimation:this.props.openAnimation,defaultOpenKeys:this.props.defaultOpenKeys,overflowedIndicator:n,style:{width:"100%"}},a))}}]),t}(c.a.Component);function R(e){return c.a.createElement(B,{backgroundColor:"red",set_nn:e.set_nn,mode:"horizontal",openAnimation:"slide-up"})}function L(e){var t=["Stream","File with record"],a=Object(y.a)((function(e){return{formControl:{display:"flex",margin:"auto",width:"40%",marginTop:"6%"}}}))(),n=c.a.useState({type:"",name:""}),i=Object(v.a)(n,2),r=i[0],s=i[1],l=["enet-coco","yolov3/coco","yolov3/drone","yolov3/openimages","yolov3-tiny/coco","yolov3-tiny/drone","yolov3-spp/coco","yolov3-spp/drone/spp1","yolov3-spp/drone/spp3","yolov3-spp-slim/drone/prune_0.5","yolov3-spp-slim/drone/prune_0.5_0.5_0.7","yolov3-spp-slim/drone/prune_0.9","yolov3-spp-slim/drone/prune_0.95","ttfnet/coco/ttfnet_d53_1x","ttfnet/coco/ttfnet_d53_2x","ttfnet/coco/ttfnet_r18_1x","ttfnet/coco/ttfnet_r18_2x","ttfnet/coco/ttfnet_r34_2x"],o=["Upload file first","Choose detector first","Number of frames must be natural number","Link is incorrect"],m=c.a.useState(""),p=Object(v.a)(m,2),u=p[0],h=p[1],d=c.a.useState(""),_=Object(v.a)(d,2),g=_[0],f=_[1],O=function(){return""!==g?(!0,!0):o[1]};return 0===e.page?c.a.createElement("div",null,c.a.createElement("div",{className:a.formControl},c.a.createElement(R,{set_nn:f})),c.a.createElement(w.a,{className:a.formControl},c.a.createElement(C.a,{value:l[g-1],margin:"normal",helperText:"Chosen type of detector",InputProps:{readOnly:!0}})),c.a.createElement(w.a,{className:a.formControl},c.a.createElement(b.a,{htmlFor:"type-helper"},"Type of video"),c.a.createElement(x.a,{value:r.type,onChange:function(e){h(""),s({type:e.target.value,name:t[e.target.value]})},inputProps:{name:"type",id:"type-helper"}},c.a.createElement(E.a,{value:""},c.a.createElement("em",null,"None")),c.a.createElement(E.a,{value:0},t[0]),c.a.createElement(E.a,{value:1},t[1])),c.a.createElement(k.a,null,"Choose type of video")),c.a.createElement(F,{set_im_src:e.set_im_src,set_page_to_1:e.set_page_to_1,width:"40%",values:r,types:t,nn:l[g-1],default_check:O,marginTop:"4%",seterror_mesage:h,error_mesage:u,error_mesage_strings:o}),c.a.createElement(P,{set_im_src:e.set_im_src,set_page_to_1:e.set_page_to_1,width:"40%",values:r,types:t,nn:l[g-1],default_check:O,marginTop:"6%",seterror_mesage:h,error_mesage:u,error_mesage_strings:o})):null}var A=a(94),q=a(4),U=a(9),J=a(378),K=a(379),G=a(376),V=a(374),X=a(385),Y=a(375),Q=a(387),Z=a(377),$=a(72),ee=a(369),te=a(386),ae=a(330),ne=a(388),ie=a(380),re=a(381),se=a(173),le=a.n(se);function oe(e,t,a){return t[a]<e[a]?-1:t[a]>e[a]?1:0}var ce=[{id:"x0",numeric:!0,disablePadding:!1,label:"x0"},{id:"y0",numeric:!0,disablePadding:!1,label:"y0"},{id:"x1",numeric:!0,disablePadding:!1,label:"x1"},{id:"y1",numeric:!0,disablePadding:!1,label:"y1"}];function me(e){var t=e.classes,a=e.onSelectAllClick,n=e.order,i=e.orderBy,r=e.numSelected,s=e.rowCount,l=e.onRequestSort;return c.a.createElement(V.a,null,c.a.createElement(Y.a,null,c.a.createElement(G.a,{padding:"checkbox"},c.a.createElement(te.a,{indeterminate:r>0&&r<s,checked:r===s,onChange:a,inputProps:{"aria-label":"select all desserts"}})),ce.map((function(e){return c.a.createElement(G.a,{key:e.id,align:e.numeric?"right":"left",padding:e.disablePadding?"none":"default",sortDirection:i===e.id&&n},c.a.createElement(Q.a,{active:i===e.id,direction:n,onClick:(a=e.id,function(e){l(e,a)})},e.label,i===e.id?c.a.createElement("span",{className:t.visuallyHidden},"desc"===n?"sorted descending":"sorted ascending"):null));var a}))))}var pe=function(e){var t=Object(y.a)((function(e){return{root:{paddingLeft:e.spacing(2),paddingRight:e.spacing(1)},highlight:"light"===e.palette.type?{color:e.palette.secondary.main,backgroundColor:Object(U.d)(e.palette.secondary.light,.85)}:{color:e.palette.text.primary,backgroundColor:e.palette.secondary.dark},spacer:{flex:"1 1 100%"},actions:{color:e.palette.text.secondary},title:{flex:"0 0 auto"}}}))(),a=e.numSelected;return c.a.createElement(Z.a,{className:Object(q.a)(t.root,Object(A.a)({},t.highlight,a>0))},c.a.createElement("div",{className:t.title},a>0?c.a.createElement($.a,{color:"inherit",variant:"subtitle1"},a," selected"):c.a.createElement($.a,{variant:"h6",id:"tableTitle"},"Lines")),c.a.createElement("div",{className:t.spacer}),c.a.createElement("div",{className:t.actions},a>0?c.a.createElement(ne.a,{title:"Delete"},c.a.createElement(ae.a,{"aria-label":"delete",onClick:e.del},c.a.createElement(le.a,null))):c.a.createElement("div",null)))};function ue(e){var t=Object(y.a)((function(e){return{root:{width:"100%",marginTop:e.spacing(3)},paper:{width:"100%",marginBottom:e.spacing(2)},table:{minWidth:750},tableWrapper:{overflowX:"auto"},visuallyHidden:{border:0,clip:"rect(0 0 0 0)",height:1,margin:-1,overflow:"hidden",padding:0,position:"absolute",top:20,width:1}}}))(),a=c.a.useState("asc"),n=Object(v.a)(a,2),i=n[0],r=n[1],s=c.a.useState("x0"),l=Object(v.a)(s,2),o=l[0],m=l[1],p=c.a.useState([]),u=Object(v.a)(p,2),h=u[0],d=u[1],_=c.a.useState(0),g=Object(v.a)(_,2),f=g[0],b=g[1],E=c.a.useState(!1),k=Object(v.a)(E,2),w=k[0],x=k[1],O=c.a.useState(5),C=Object(v.a)(O,2),S=C[0],j=C[1],F=e.rows,I=S-Math.min(S,F.length-f*S);return c.a.createElement("div",{className:t.root},c.a.createElement(ee.a,{className:t.paper},c.a.createElement(pe,{numSelected:h.length,del:function(t){e.del_marked(h),d([])}}),c.a.createElement("div",{className:t.tableWrapper},c.a.createElement(J.a,{className:t.table,"aria-labelledby":"tableTitle",size:w?"small":"medium"},c.a.createElement(me,{classes:t,numSelected:h.length,order:i,orderBy:o,onSelectAllClick:function(t){if(t.target.checked){var a=F.map((function(e){return e.name}));return d(a),void e.load_marked(a)}d([]),e.load_marked([])},onRequestSort:function(e,t){r(o===t&&"desc"===i?"asc":"desc"),m(t)},rowCount:F.length}),c.a.createElement(K.a,null,function(e,t){var a=e.map((function(e,t){return[e,t]}));return a.sort((function(e,a){var n=t(e[0],a[0]);return 0!==n?n:e[1]-a[1]})),a.map((function(e){return e[0]}))}(F,function(e,t){return"desc"===e?function(e,a){return oe(e,a,t)}:function(e,a){return-oe(e,a,t)}}(i,o)).slice(f*S,f*S+S).map((function(t,a){var n,i=(n=t.name,-1!==h.indexOf(n)),r="enhanced-table-checkbox-".concat(a);return c.a.createElement(Y.a,{hover:!0,onClick:function(a){return function(t,a){var n=h.indexOf(a),i=[];-1===n?i=i.concat(h,a):0===n?i=i.concat(h.slice(1)):n===h.length-1?i=i.concat(h.slice(0,-1)):n>0&&(i=i.concat(h.slice(0,n),h.slice(n+1))),d(i),e.load_marked(i)}(0,t.name)},role:"checkbox","aria-checked":i,tabIndex:-1,key:t.name,selected:i},c.a.createElement(G.a,{padding:"checkbox"},c.a.createElement(te.a,{checked:i,inputProps:{"aria-labelledby":r}})),c.a.createElement(G.a,{align:"right"},t.x0),c.a.createElement(G.a,{align:"right"},t.y0),c.a.createElement(G.a,{align:"right"},t.x1),c.a.createElement(G.a,{align:"right"},t.y1))})),I>0&&c.a.createElement(Y.a,{style:{height:49*I}},c.a.createElement(G.a,{colSpan:6}))))),c.a.createElement(X.a,{rowsPerPageOptions:[5,10,25],component:"div",count:F.length,rowsPerPage:S,page:f,backIconButtonProps:{"aria-label":"previous page"},nextIconButtonProps:{"aria-label":"next page"},onChangePage:function(e,t){b(t)},onChangeRowsPerPage:function(e){j(+e.target.value),b(0)}})),c.a.createElement(ie.a,{control:c.a.createElement(re.a,{checked:w,onChange:function(e){x(e.target.checked)}}),label:"Dense padding"}))}var he=function(e){function t(e){var a;return Object(n.a)(this,t),(a=Object(r.a)(this,Object(s.a)(t).call(this,e))).W=a.props.W,a.H=a.props.H,a.k=0,a.line_width=1,a.canvasRef=c.a.createRef(),a.handleClick=a.handleClick.bind(Object(M.a)(a)),a}return Object(l.a)(t,e),Object(i.a)(t,[{key:"handleClick",value:function(e){e.preventDefault();var t=this.canvasRef.current,a=(t.getContext("2d"),t.getBoundingClientRect()),n=e.clientX-a.left,i=e.clientY-a.top;if(0===this.k)this.props.setPoint(n,i);else{var r=Math.round((n+this.props.point[0])/2),s=Math.round((i+this.props.point[1])/2),l=n-this.props.point[0],o=i-this.props.point[1],c=1,m=1;0!==o?c=-l/o:m=0,n>this.props.point[0]&&c<0&&(c=-c,m=-m),n<this.props.point[0]&&c>0&&(c=-c,m=-m),n===this.props.point[0]&&m<0&&(c=-c,m=-m);var p=Math.sqrt(m*m+c*c);m=Math.round(50*m/p),c=Math.round(50*c/p);var u=Math.round(m+r),h=Math.round(c+s);this.props.setLine(n,i,r,s,u,h)}this.k++,this.k=this.k%2}},{key:"drawline",value:function(e,t){e.beginPath(),e.arc(t.x0,t.y0,this.line_width,0,2*Math.PI,!0),e.arc(t.x1,t.y1,this.line_width,0,2*Math.PI,!0),e.fill(),e.beginPath(),e.lineWidth=2*this.line_width,e.moveTo(t.x0,t.y0),e.lineTo(t.x1,t.y1),e.stroke()}},{key:"drawarrow",value:function(e,t){e.beginPath(),e.arc(t.x0,t.y0,this.line_width,0,2*Math.PI,!0),e.arc(t.x1,t.y1,this.line_width,0,2*Math.PI,!0),e.fill(),e.beginPath(),e.lineWidth=2*this.line_width,e.moveTo(t.x0,t.y0),e.lineTo(t.x1,t.y1),e.stroke();var a=Math.atan2(t.y1-t.y0,t.x1-t.x0),n=10*Math.cos(a)+t.x1,i=10*Math.sin(a)+t.y1;e.moveTo(n,i),a+=1/3*(2*Math.PI),n=10*Math.cos(a)+t.x1,i=10*Math.sin(a)+t.y1,e.lineTo(n,i),a+=1/3*(2*Math.PI),n=10*Math.cos(a)+t.x1,i=10*Math.sin(a)+t.y1,e.lineTo(n,i),e.closePath(),e.fill()}},{key:"drawpoint",value:function(e){e.beginPath(),e.arc(this.props.point[0],this.props.point[1],this.line_width,0,2*Math.PI,!0),e.fill()}},{key:"draw",value:function(){if(null!==this.canvasRef.current){var e=this.canvasRef.current,t=e.getContext("2d");t.clearRect(0,0,e.width,e.height);var a=this.props.color_lines;t.fillStyle=a,t.strokeStyle=a;for(var n=0;n<this.props.lines.length;n++)this.drawline(t,this.props.lines[n]),this.drawarrow(t,this.props.perp[n]);null!==this.props.point&&this.drawpoint(t),a=this.props.color_lines_marked,t.fillStyle=a,t.strokeStyle=a;for(var i=0;i<this.props.marked.length;i++)this.drawline(t,this.props.lines[this.props.marked[i]]),this.drawarrow(t,this.props.perp[this.props.marked[i]])}}},{key:"render",value:function(){return this.draw(),c.a.createElement("div",{width:this.W,height:this.H},c.a.createElement("img",{alt:"First frame",src:this.props.im_src,width:this.W,height:this.H,style:{zIndex:1,display:"flex",visibility:"visible",left:0,right:0,margin:"auto",position:"absolute",align:"center"}}),c.a.createElement("canvas",{ref:this.canvasRef,style:{position:"relative",display:"flex",margin:"auto",zIndex:20},width:this.W,height:this.H,onClick:this.handleClick}))}}]),t}(c.a.Component);function de(e,t,a,n,i){return{name:e,x0:t,y0:a,x1:n,y1:i}}var _e=function(e){function t(e){var i;return Object(n.a)(this,t),(i=Object(r.a)(this,Object(s.a)(t).call(this,e))).setPoint=function(e,t){i.setState({point:[e,t]})},i.setLine=function(e,t,a,n,r,s){i.setState({lines:i.state.lines.concat(de(i.state.line_num,Math.ceil(i.state.point[0]),Math.ceil(i.state.point[1]),Math.ceil(e),Math.ceil(t))),perp:i.state.perp.concat(de(i.state.line_num,Math.ceil(a),Math.ceil(n),Math.ceil(r),Math.ceil(s))),line_num:i.state.line_num+1,point:null})},i.load_marked=function(e){i.setState({marked:i.state.marked=e.slice()})},i.del_marked=function(e){i.setState({marked:i.state.marked=[]});for(var t=i.state.lines.slice(),a=i.state.perp.slice(),n=0;n<e.length;n++){t.splice(e[n],1),a.splice(e[n],1);for(var r=n+1;r<e.length;r++)e[r]>e[n]&&e[r]--}for(var s=0;s<t.length;s++)t[s].name=s,a[s].name=s;i.setState({lines:i.state.lines=t.slice(),perp:i.state.perp=a.slice(),line_num:t.length})},i.postreq=function(){a(69)({method:"post",url:"",data:{lines:JSON.stringify(i.state.lines)}}).then((function(e){console.log(e)})).catch((function(e){return console.warn(e)}))},i.handleClick=function(e){e.preventDefault(),i.postreq(),i.props.set_num_of_lines(i.state.line_num),i.props.set_streamurl(),i.props.set_page_to_2()},i.run_button="Run",i.W=800,i.H=450,i.state={point:null,line_num:0,lines:[],perp:[],marked:[],stage:0},i}return Object(l.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){return 1===this.props.page?c.a.createElement("div",null,c.a.createElement(he,{color_lines:this.props.color_lines,color_lines_marked:this.props.color_lines_marked,setPoint:this.setPoint,setLine:this.setLine,point:this.state.point,lines:this.state.lines,marked:this.state.marked,im_src:this.props.im_src,perp:this.state.perp,W:this.W,H:this.H}),c.a.createElement(O.a,{variant:"outlined",color:"primary",onClick:this.handleClick,style:{width:"50%",position:"relative",display:"flex",margin:"auto",marginTop:"25px",marginBottom:"25px"}},this.run_button),c.a.createElement(ue,{load_marked:this.load_marked,rows:this.state.lines,del_marked:this.del_marked})):null}}]),t}(c.a.Component);function ge(e){return!0===e.show_button?c.a.createElement(O.a,{variant:"outlined",color:"primary",onClick:function(t){!function(){var t=a(69);t({method:"post",url:"/get_stats"}).then((function(a){200===a.status&&(e.set_sim(),t({method:"post",url:"/get_rec_gates"}).then((function(t){200===t.status&&e.set_rec_gates()})).catch((function(e){return console.warn(e)})))})).catch((function(e){return console.warn(e)}))}(),e.set_page_to_3()},style:{width:"50%",position:"relative",display:"flex",margin:"auto",marginTop:"25px",marginBottom:"25px"}},"Show stats"):null}var fe=a(370),ve=a(329),ye=a(382),be=a(383),Ee=a(174),ke=a.n(Ee),we=a(175),xe=a.n(we),Oe=["#0000FF","#00FF00","#FF0000","#00FFFF","#FF00FF","#FFFF00","#000000","#FFFFFF","#8080FF","#FF8080","#80FF80","#808080"],Ce=function(e){function t(){var e,a;Object(n.a)(this,t);for(var i=arguments.length,l=new Array(i),o=0;o<i;o++)l[o]=arguments[o];return(a=Object(r.a)(this,(e=Object(s.a)(t)).call.apply(e,[this].concat(l)))).state={open:{}},a.handleClick=function(e){return function(){console.log(e),a.setState(Object(A.a)({},e,!a.state[e]))}},a}return Object(l.a)(t,e),Object(i.a)(t,[{key:"render",value:function(){var e=this,t=this.props,a=t.lists;t.classes;return c.a.createElement("div",null,c.a.createElement(fe.a,{component:"nav"},a.map((function(t){var a=t.key,n=t.label,i=t.items,r=e.state[a]||!1;return c.a.createElement("div",{key:a},c.a.createElement(ve.a,{button:!0,onClick:e.handleClick(a)},c.a.createElement(ye.a,{style:{color:Oe[a%12]},inset:!0,primary:n}),r?c.a.createElement(ke.a,null):c.a.createElement(xe.a,null)),c.a.createElement(be.a,{in:r,timeout:"auto",unmountOnExit:!0},c.a.createElement(fe.a,{component:"div",disablePadding:!0},i.map((function(e){var t=e.key,a=e.label;return c.a.createElement(ve.a,{key:t,button:!0},c.a.createElement(ye.a,{inset:!0,primary:a}))})))))}))))}}]),t}(c.a.Component),Se=function(e){function t(e){var i;return Object(n.a)(this,t),(i=Object(r.a)(this,Object(s.a)(t).call(this,e))).set_response_data=function(e){i.setState({response_data:i.state.response_data=e})},i.set_lines_data=function(e){i.setState({lines_data:i.state.lines_data=e})},i.set_exit=function(e){i.setState({exit:e})},i.set_show_button=function(){i.setState({show_button:!0})},i.getreq=function(){a(69)({method:"get",url:"/get_current_det"}).then((function(e){200===e.status&&i.set_response_data(e.data)})).catch((function(e){return console.warn(e)}))},i.timer=function(){i.getreq();var e=[];for(var t in i.state.response_data){if("exit"!==t){var a={},n=i.state.response_data[t];for(var r in a.key=t,a.label="Line "+t+":"+n[0],a.items=[],n=n[1]){var s={};s.key=r,s.label=n[r][0]+":"+n[r][1],a.items.push(s)}e.push(a)}"exit"===t&&i.set_exit(i.state.response_data[t])}i.set_lines_data(e),!0===i.state.exit&&(clearInterval(i.state.intervalId),i.set_show_button())},i.timer_time=1e3,i.state={response_data:{},intervalId:null,lines_data:[],exit:!1,show_button:!1},i}return Object(l.a)(t,e),Object(i.a)(t,[{key:"componentDidUpdate",value:function(e){if(this.props.page!==e.page&&2===this.props.page){var t=setInterval(this.timer,this.timer_time);this.setState({intervalId:t})}}},{key:"render",value:function(){var e=this;if(2===this.props.page){var t=parseInt(.5*this.props.width)+"px",a=parseInt(.5*this.props.height)+"px";return c.a.createElement(c.a.Fragment,null,c.a.createElement("div",{style:{width:t,height:a,position:"relative",display:"flex",margin:"auto",backgroundColor:"black"}},c.a.createElement("img",{style:{width:t,height:a,zIndex:1,display:"flex",visibility:"visible"},alt:"stream",id:"video",src:this.props.stream_url,onError:function(t){t.target.onerror=null,t.target.src=e.props.Background}})),c.a.createElement(ge,{show_button:this.state.show_button,set_page_to_3:this.props.set_page_to_3,set_sim:this.props.set_sim,set_rec_gates:this.props.set_rec_gates,set_empty_frame:this.props.set_empty_frame}),c.a.createElement(Ce,{lists:this.state.lines_data}))}return null}}]),t}(c.a.Component),je=a(176),Fe=a.n(je),Ie=a(177),Ne=a.n(Ie),Te=["Stats vizualization","Clusterized clear ways","Frame without detectable objets"];function Pe(e){if(3===e.page){var t=parseInt(.2*e.width)+"px",a=parseInt(.2*e.height)+"px",n=parseInt(.5*e.width)+"px",i=parseInt(.5*e.height)+"px",r=[{src:e.s_im_0,thumbnail:e.s_im_0,thumbnailWidth:t,thumbnailHeight:a,caption:"All detections"},{src:e.s_im_1,thumbnail:e.s_im_1,thumbnailWidth:t,thumbnailHeight:a,caption:"Trajectories"},{src:e.s_im_2,thumbnail:e.s_im_2,thumbnailWidth:t,thumbnailHeight:a,caption:"Smooth and clear trajectories"},{src:e.s_im_3,thumbnail:e.s_im_3,thumbnailWidth:t,thumbnailHeight:a,caption:"Time(sec) heetmap"},{src:e.s_im_4,thumbnail:e.s_im_4,thumbnailWidth:t,thumbnailHeight:a,caption:"Trajectory length(pixels) heetmap"}];return c.a.createElement(c.a.Fragment,null,c.a.createElement($.a,{variant:"h2",gutterBottom:!0,align:"center"},Te[0]),c.a.createElement("div",{style:{display:"block",minHeight:"1px",width:"100%",overflow:"auto"}},c.a.createElement(Fe.a,{enableImageSelection:!1,images:r})),c.a.createElement($.a,{variant:"h2",gutterBottom:!0,align:"center",style:{display:"block",marginTop:"50px"}},Te[1]),c.a.createElement(Ne.a,{image:{src:e.rec_gates,alt:Te[1],className:"img",style:{display:"flex",margin:"auto",width:n,height:i}},zoomImage:{src:e.rec_gates,alt:Te[1]}}))}return null}var Me=a(32),We=a.n(Me),He=f.a,De=_.a,ze=(He[500],De[500],Object(u.a)({spacing:4,palette:{type:"light",primary:He,secondary:De},status:{danger:"orange"}})),Be=function(e){function t(e){var a;return Object(n.a)(this,t),(a=Object(r.a)(this,Object(s.a)(t).call(this,e))).set_button_to_stats=function(){a.setState({button_to_stats:!0})},a.set_page_to_1=function(){a.setState({page:1})},a.set_page_to_2=function(){a.setState({page:2})},a.set_page_to_3=function(){a.setState({page:3})},a.set_im_src=function(){a.setState({im_src:"http://127.0.0.1:5000/get_image"})},a.set_streamurl=function(){a.setState({stream_url:"http://127.0.0.1:5000/get_stream"})},a.set_sim=function(){a.setState({s_im_0:"http://127.0.0.1:5000/get_stats0",s_im_1:"http://127.0.0.1:5000/get_stats1",s_im_2:"http://127.0.0.1:5000/get_stats2",s_im_3:"http://127.0.0.1:5000/get_stats3",s_im_4:"http://127.0.0.1:5000/get_stats4"})},a.set_empty_frame=function(){a.setState({empty_frame:"http://127.0.0.1:5000/get_empty"})},a.set_rec_gates=function(){a.setState({rec_gates:"http://127.0.0.1:5000/get_rec"})},a.set_num_of_lines=function(e){a.setState({num_of_lines:e})},a.updateWindowDimensions=function(){a.setState({width:window.innerWidth,height:window.innerHeight})},a.state={width:0,height:0,page:0,im_src:We.a,stream_url:"",num_of_lines:0,button_to_stats:!0,s_im_0:We.a,s_im_1:We.a,s_im_2:We.a,s_im_3:We.a,s_im_4:We.a,empty_frame:We.a,rec_gates:We.a},a}return Object(l.a)(t,e),Object(i.a)(t,[{key:"componentDidMount",value:function(){this.updateWindowDimensions(),window.addEventListener("resize",this.updateWindowDimensions)}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.updateWindowDimensions)}},{key:"render",value:function(){return console.log(He[500]),c.a.createElement(h.a,{theme:ze},c.a.createElement("div",null,c.a.createElement(L,{page:this.state.page,set_page_to_1:this.set_page_to_1,set_im_src:this.set_im_src,width:this.state.width,height:this.state.height}),c.a.createElement(_e,{page:this.state.page,set_page_to_2:this.set_page_to_2,im_src:this.state.im_src,color_lines:He[500],color_lines_marked:De[500],set_streamurl:this.set_streamurl,width:this.state.width,height:this.state.height,set_num_of_lines:this.set_num_of_lines}),c.a.createElement(Se,{Background:We.a,page:this.state.page,set_page_to_3:this.set_page_to_3,stream_url:this.state.stream_url,set_sim:this.set_sim,set_rec_gates:this.set_rec_gates,set_empty_frame:this.set_empty_frame,width:this.state.width,height:this.state.height,num_of_lines:this.state.num_of_lines}),c.a.createElement(Pe,{page:this.state.page,s_im_0:this.state.s_im_0,s_im_1:this.state.s_im_1,s_im_2:this.state.s_im_2,s_im_3:this.state.s_im_3,s_im_4:this.state.s_im_4,rec_gates:this.state.rec_gates,empty_frame:this.state.empty_frame,width:this.state.width,height:this.state.height})))}}]),t}(c.a.Component);p.a.render(c.a.createElement(Be,null),document.getElementById("root"))}},[[196,1,2]]]);
//# sourceMappingURL=main.14498e30.chunk.js.map