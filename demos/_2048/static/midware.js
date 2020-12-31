/**
 * Created by Eleven on 2017/1/3.
 */
function setBackground(num){
    switch (num){
        case 2:return "#eee4da";break;
        case 4:return "#ede0c8";break;
        case 8:return "#f2b179";break;
        case 16:return "#f59563";break;
        case 32:return "#f67c5f";break;
        case 64:return "#f65e3b";break;
        case 128:return "#edcf72";break;
        case 256:return "#edcc61";break;
        case 512:return "#9c0";break;
        case 1024:return "#33b5e5";break;
        case 2048:return "#09c";break;
        case 4096:return "#a6c";break;
        case 8192:return "#93c";break;
    }
    return 'black'
}
function setColor(num){
    if(num<=4) return '#776e65'
    return '#fff'
}
function setFont(num){
    if(num<=8){ return '0.5rem'}
    else if(num<128){return '0.46rem'}
    else if(num<1024){return '0.4rem'}
    return '0.34rem'

}
function canMoveLeft( board ){
    var nowR=$('.number-cell:last').data('row')+1
    var nowC=$('.number-cell:last').data('col')+1
    for( var i = 0 ; i < nowR ; i ++ )
        for( var j =1; j <nowC ; j ++ )
            if( board[i][j] != 0 )
                if( board[i][j-1] == 0 || board[i][j-1] == board[i][j] )
                    return true;

    return false;
}

function canMoveRight( board ){
    var nowR=$('.number-cell:last').data('row')+1
    var nowC=$('.number-cell:last').data('col')+1
    var RnowC=nowC-2
    for( var i = 0 ; i < nowR ; i ++ )
        for( var j = RnowC; j >= 0 ; j -- )
            if( board[i][j] != 0 )
                if( board[i][j+1] == 0 || board[i][j+1] == board[i][j] )
                    return true;

    return false;
}

function canMoveUp( board ){
    var nowR=$('.number-cell:last').data('row')+1
    var nowC=$('.number-cell:last').data('col')+1
    for( var j = 0 ; j < nowC ; j ++ )
        for( var i = 1 ; i < nowR ; i ++ )
            if( board[i][j] != 0 )
                if( board[i-1][j] == 0 || board[i-1][j] == board[i][j] )
                    return true;

    return false;
}

function canMoveDown( board ){
    var nowR=$('.number-cell:last').data('row')+1
    var nowC=$('.number-cell:last').data('col')+1
    var DnowR=nowR-2
    for( var j = 0 ; j < nowC ; j ++ )
        for( var i = DnowR ; i >= 0 ; i -- )
            if( board[i][j] != 0 )
                if( board[i+1][j] == 0 || board[i+1][j] == board[i][j] )
                    return true;

    return false;
}


function noBlockHorizon(row,col1,col2,board){
    for(i=col1+1;i<col2;i++){
        if(board[row][i]!=0){
            return false
        }
    }
    return true
}
function noBlockVertical(col,row1,row2,board){
    for(i=row1+1;i<row2;i++){
        if(board[i][col]!=0){
            return false
        }
    }
    return true
}

