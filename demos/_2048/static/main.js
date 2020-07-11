/**
 * Created by Eleven on 2017/1/3.
 */
var board=new Array() //生成一维数组
var score=0;
var isTraining = false;
var isAiPlaying = false;
var startX,startY,endX,endY,distantX,distantY;

$(function () {
    //初始化4x4方格
     init(4,4)
    var personal=$('#personal')
    var gameover=$('#gameover')
    var container=$('#container')[0]
    //pc端
    $('body').click(function (e) {
        var target= e.target
        if(target==container||target.parentNode==container){
            gameover.hide()
            personal.hide()
        }
        //手机端
        $('body').bind('touchastart', function (e) {
            var target= e.target
        }).bind('touchend', function (e) {
            var target= e.target
            if(target==container||target.parentNode==container){
                gameover.hide()
                personal.hide()
            }
        })
    })
})
function reStart(){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1
    var nowC=nowNumberCell.data('col')+1
    init(nowR,nowC)
    $('#gameover').hide();
    if(isAiPlaying){
        if(interval == null) {
            interval = setInterval(ai_play, 500);
        }

    }

}

//初始化函数
function init(w,h){
    var wrap=$('.grid');
    wrap.empty();
    var wrapWidth=1.2*w+0.2
    var wrapHeight=1.2*h+0.2
    wrap.css('width',wrapWidth+'rem')
    wrap.css('height',wrapHeight+'rem')
    var showScore=$('#score')
    for(var i=0;i<h;i++){
        for(var j=0;j<w;j++){
          var gridCell=$('<div class="grid-cell"></div>')
            gridCell.attr('id','grid-cell-'+i+"-"+j)
            gridCell.css('top',setPos(i,null))
            gridCell.css('left',setPos(null,j))
            gridCell.appendTo(wrap)

        }
    }
    for(var m=0;m<w;m++){
        board[m]=new Array()
        for(var k=0;k<h;k++){
            board[m][k]=0
        }
        }
    updateBoard(w,h)
    generateNumber()
    generateNumber()
    score=0;
    showScore.html(score)
}

//判断页面内是否还有空格
function noPlace(board){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1
    var nowC=nowNumberCell.data('col')+1
    for(i=0;i<nowR;i++){
        for(j=0;j<nowC;j++){
            if(board[i][j]==0){
                return false
            }
        }
    }
    return true
}

//设置定位
function setPos(i,j){
    if(i){
        return i*1.2+0.2+'rem'
    }
    return j*1.2+0.2+'rem'
}

function collectTrainingData(){
    isTraining = !isTraining;
    if(isTraining) {
        $("#collectBtn").html("正在收集");
    }
    else {
        $("#collectBtn").html("收集完毕");
    }
}

function training() {
    var nowNumberCell = $('.number-cell:last');
    var h = nowNumberCell.data('row') + 1;
    var w = nowNumberCell.data('col') + 1;
    var data = {'height': h, 'width': w};
    $.ajax({
        type: 'POST',
        url: "/train",
        data: data,
        dataType: 'json',
        traditional: true,
        success: function (data) {
            alert('训练完毕！');
        },
        error: function (xhr, type) {
        }
    });
}

var interval;
function ai_play() {
    var nowNumberCell = $('.number-cell:last');
    var h = nowNumberCell.data('row') + 1;
    var w = nowNumberCell.data('col') + 1;
    var data = {};
    isTraining = false;
    data['height'] = h;
    data['width'] = w;
    data['array'] = JSON.stringify(collect_cells(h, w));
    $.ajax({
        type: 'POST',
        url: "/ai",
        data: data,
        dataType: 'json',
        traditional: true,
        success: function (message) {
            move(parseInt(message.key) + 37)
        },
        error: function (xhr, type) {
        }
            });
}

function AI() {
    if(isAiPlaying) {
        $("#aiBtn").html("人工模式");
        isAiPlaying = false;
        if(interval != null) {
            clearInterval(interval);
        }
    }
    else {
        $("#aiBtn").html("AI模式");
        isAiPlaying = true;
        interval = setInterval(ai_play, 500);
    }
}

//每次移动后更新视图及数据的函数
function updateBoard(w,h){
    $('.number-cell').remove()
    for(i=0;i<h;i++){
       for(j=0;j<w;j++){
           $('.grid').append('<div class="number-cell" id="number-cell-'+i+'-'+j+'"></div>')
           var nowNumberCell=$('#number-cell-'+i+"-"+j)
           nowNumberCell.css('top',setPos(i,null))
           nowNumberCell.css('left',setPos(null,j))
           nowNumberCell.attr('data-row',i)
           nowNumberCell.attr('data-col',j)
           if(board[i][j]==0){
               nowNumberCell.css('width','0rem')
               nowNumberCell.css('height','0rem')
               //nowNumberCell.css('top',setOffPos(i,null)+0.5+'rem')
               //nowNumberCell.css('left',setOffPos(null,j)+0.5+'rem')
           }else{
               //nowNumberCell.css('top',setPos(i,null))
               //nowNumberCell.css('left',setPos(null,j))
               nowNumberCell.css('width','1rem')
               nowNumberCell.css('height','1rem')
               nowNumberCell.css('background',setBackground(board[i][j]))
               nowNumberCell.css('color',setColor(board[i][j]))
               nowNumberCell.css('fontSize',setFont(board[i][j]))
               nowNumberCell.text(board[i][j])
           }
       }
   }
    var showScore=$('#score')
    showScore.html(score)
}

function collect_cells(h, w){
    var cell_array = [];
    for (var i = 0; i < h; i++) {
        for (var j = 0; j < w; j++) {
            var val = $('#number-cell-' + i + "-" + j).html();
            if (typeof val == "undefined" || val == null || val == "") {
                val = "0";
            }
            cell_array[i * w + j] = parseInt(val);
        }
    }
    return cell_array;
}
function move(keyCode) {
        if (isTraining) {
            var data = {};
            var nowNumberCell = $('.number-cell:last');
            var h = nowNumberCell.data('row') + 1;
            var w = nowNumberCell.data('col') + 1;
            var cell_array = collect_cells(h, w);
            data['array'] = JSON.stringify(cell_array);
            data['height'] = h;
            data['width'] = w;
        }

    switch(keyCode){
        //left
        case 37:
            if (moveToLeft()) {
                setTimeout(generateNumber,160);
                setTimeout(isGameOver,200);
                if(isTraining) {
            data['key'] = 0;
            $.ajax({
                type: 'POST',
                url: "/record",
                data: data,
                dataType: 'json',
                traditional: true,
                success: function (data) {
                },
                error: function (xhr, type) {
                }
            });
        }
            return true;
            }

            break;
        //up
        case 38:
            if(moveToUp()){
                setTimeout(generateNumber,160)
                setTimeout(isGameOver,200)
               if(isTraining) {
                    data['key'] = 1;
                    $.ajax({
                        type: 'POST',
                        url: "/record",
                        data: data,
                        dataType: 'json',
                        traditional: true,
                        success: function (data) {
                        },
                        error: function (xhr, type) {
                        }
                    });
                }
            return true;
            }
            break;
        //right
        case 39:
            if(moveToRight()){
                setTimeout(generateNumber,160)
                setTimeout(isGameOver,200)
                if(isTraining) {
            data['key'] = 2;
            $.ajax({
                type: 'POST',
                url: "/record",
                data: data,
                dataType: 'json',
                traditional: true,
                success: function (data) {
                },
                error: function (xhr, type) {
                }
            });
        }
                return true;
            }

            break;
        //down
        case 40:
            if(moveToDown()){
                setTimeout(generateNumber,160)
                setTimeout(isGameOver,200)
                if(isTraining) {
            data['key'] = 3;
            $.ajax({
                type: 'POST',
                url: "/record",
                data: data,
                dataType: 'json',
                traditional: true,
                success: function (data) {
                },
                error: function (xhr, type) {
                }
            });
        }
            return true;
            }
            break;
        default :
            return false;
    }
}

//键盘事件
$(document).keydown(function (event) {
        move(event.keyCode);
    })
//屏幕滑动事件
document.addEventListener('touchstart', function (event) {
    startX=event.touches[0].pageX;
    startY=event.touches[0].pageY
})
document.addEventListener('touchmove', function (event) {
    //endX=event.changedTouches[0].pageX;
    //endY=event.changedTouches[0].pageY;
    //distantX=Math.abs(startX-endX);//???????????
    //distantY=Math.abs(startY-endY);//????????????
    event.preventDefault()
})
document.addEventListener('touchend', function (event) {
   endX=event.changedTouches[0].pageX;
   endY=event.changedTouches[0].pageY;
   distantX=Math.abs(startX-endX);//???????????
   distantY=Math.abs(startY-endY);//????????????
    if(distantX>30||distantY>30) {

        if (distantX > distantY) {
            //?ж????????
            if (startX > endX) {
                //?????
                if (moveToLeft()) {
                    setTimeout(generateNumber, 160)
                    setTimeout(isGameOver, 200)
                }
            }
            else {
                //???????
                if (moveToRight()) {
                    setTimeout(generateNumber, 160)
                    setTimeout(isGameOver, 200)
                }
            }
        }
        else {
            //?ж?????????
            if (startY > endY) {
                //???????
                if (moveToUp()) {
                    setTimeout(generateNumber, 160)
                    setTimeout(isGameOver, 200)
                }

            }
            else {
                //???????
                if (moveToDown()) {
                    setTimeout(generateNumber, 160)
                    setTimeout(isGameOver, 200)
                }
            }
        }
    }
})
function moveToLeft() {
    var nowNumberCell = $('.number-cell:last')
    var nowR = nowNumberCell.data('row') + 1;
    var nowC = nowNumberCell.data('col') + 1;
    var LnowC=nowC-1
    if (!canMoveLeft( board ))
        return false
    //moveLeft
    for (var i = 0; i < nowR; i++) {
        for (var j = 0; j<LnowC; j++) {
                for (var k = j+1; k < nowC; k++) {
                    if (board[i][k] != 0) {
                        if ( noBlockHorizon(i, j, k, board) && board[i][j] == 0) {
                            //move
                            moveAnimation(i, k, i, j);
                            board[i][j] = board[i][k];
                            board[i][k] = 0;
                            //continue;
                        }
                        else if (board[i][k] == board[i][j] && noBlockHorizon(i, j, k, board)) {
                            moveBigAnimation(i, k, i, j);
                            board[i][j] *= 2;
                            score+=board[i][j]
                            board[i][k] = 0;
                            j++
                        }
                        //continue;
                    }
                }
            }
    }
    function updateView(){
        updateBoard(nowR,nowC)
    }
    setTimeout(updateView,150);
    return true
}
function moveToUp(){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1;
    var nowC=nowNumberCell.data('col')+1;
    var UnowC=nowC-1;//3
    if( !canMoveUp( board ) )
        return false;
    //moveUp
    for( var j = 0 ; j< nowC ; j ++ ) {
        for (var i = 0; i < UnowC; i++) {
                for (var k = i+1; k < nowC; k++) {
                    if (board[k][j] != 0) {
                        //j??У?i??У?k?i???????
                        if (noBlockVertical(j, i, k, board) && board[i][j] == 0) {
                            moveAnimation(k, j, i, j);
                            board[i][j] = board[k][j];
                            board[k][j] = 0;
                        }
                        else if (board[k][j] == board[i][j] && noBlockVertical(j, i, k, board)) {
                            moveBigAnimation(k, j, i, j);
                            board[i][j] *= 2;
                            score+=board[i][j]
                            board[k][j] = 0;
                            i++
                        }
                    }
                }
        }
    }
    function updateView(){
        updateBoard(nowR,nowC)
    }
    setTimeout(updateView,150)
    return true
}
function moveToRight(){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1;
    var nowC=nowNumberCell.data('col')+1;
    var RnowC=nowC-1 //3
    if( !canMoveRight( board ) )
        return false;
    //moveRight
    for( var i = 0 ; i < nowR; i ++ ){
        for( var j = RnowC ; j >0; j-- ){
                for( var k =j-1; k >=0 ; k -- ) {
                    if (board[i][k] != 0) {
                        if (noBlockHorizon(i, k, j, board) && board[i][j] == 0) {
                            //move
                            moveAnimation(i, k, i, j);
                            board[i][j] = board[i][k];
                            board[i][k] = 0;
                        }
                else if (board[i][k] == board[i][j] && noBlockHorizon(i, k, j, board)) {
                            //move
                            moveBigAnimation(i, k, i, j);
                            //add
                            board[i][j] *= 2;
                            score+=board[i][j]
                            board[i][k] = 0;
                            j--
                        }
                    }
                }
        }
    }
    function updateView(){
        updateBoard(nowR,nowC)
    }
    setTimeout(updateView,150)
    return true
}
function moveToDown(){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1
    var nowC=nowNumberCell.data('col')+1
    var RnowR=nowR-1//3
    if( !canMoveDown(board) ){
        return false;
    }
    //moveDown
    for( var j = 0 ; j < nowC; j ++ )
        for( var i = RnowR; i > 0 ; i -- ){
                for( var k=i-1; k >= 0 ; k -- ) {
                    if (board[k][j] != 0) {
                        if (noBlockVertical(j, k, i, board) && board[i][j] == 0){
                            moveAnimation(k, j, i, j);
                            board[i][j] = board[k][j];
                            board[k][j] = 0;
                        }
                else if (board[k][j] == board[i][j] && noBlockVertical(j, k, i, board)){
                            moveBigAnimation(k, j, i, j);
                            board[i][j] *= 2;
                            score+=board[i][j]
                            board[k][j] = 0;
                            i--
                        }
                    }
                }
        }
    function updateView(){
        updateBoard(nowR,nowC)
    }
    setTimeout(updateView,150);
    return true;
}
//随机生成一个数
function generateNumber(){
    if(noPlace(board)){
        return false
    }
    var nowNumberCell=$('.number-cell:last');
    var nowR=nowNumberCell.data('row')+1;
    var nowC=nowNumberCell.data('col')+1;
    var numberX=parseInt(Math.floor(nowR*Math.random()));
    var numberY=parseInt(Math.floor(nowC*Math.random()));
    //生成位置
    while(true){
        if(board[numberX][numberY]==0){
            break;
        }
        numberX=parseInt(Math.floor((nowR)*Math.random()))
        numberY=parseInt(Math.floor((nowC)*Math.random()))
    }
    //生成数字
    var randomNumber=Math.random()>0.5?2:4
    board[numberX][numberY]=randomNumber
    showNumberAnimation(numberX,numberY,randomNumber)
    return true
}
//判断游戏是否结束
function isGameOver(){
    var nowNumberCell=$('.number-cell:last')
    var nowR=nowNumberCell.data('row')+1
    var nowC=nowNumberCell.data('col')+1
    if(noPlace(board)){

       if(!canMoveLeft(board)&&!canMoveRight(board)&&!canMoveUp(board)&&!canMoveDown(board)){
           $('#gameover').show(200);
           clearInterval(interval);
       }
   }
}

function showForm(){
    $('#personal').show(500)
}
function getPerData(){
    var personal=$('#personal')
    var row=personal.find('input')[0].value;
    var col=personal.find('input')[1].value;
    if(row==null||row==0&&col==null||col==0){
        return true
    }
    init(col,row)
    personal.hide()
  $('#personal input').val('')
}



