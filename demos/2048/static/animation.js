/**
 * Created by Eleven on 2017/1/3.
 */
function showNumberAnimation(x,y,num){
    var theNumberCell=$('#number-cell-'+x+"-"+y)
    theNumberCell.css('background',setBackground(num))
    theNumberCell.css('color',setColor(num))
    theNumberCell.text(num)
    var t=x*1.2+0.2+'rem'
    var l=y*1.2+0.2+'rem'
    //theNumberCell.css('transform','scale(1)')
    theNumberCell.css('width','1rem')
    theNumberCell.css('height','1rem')
    theNumberCell.css('top',setPos(x,null))
    theNumberCell.css('left',setPos(null,y))
    //theNumberCell.css('transform','translate3d(t,l,0)')
}

function moveAnimation(fromx,fromy,tox,toy){
    var numberCell=$('#number-cell-'+fromx+'-'+fromy);
    numberCell.animate({
        top:setPos(tox,null),
        left:setPos(null,toy)
    },150)

}
function moveBigAnimation(fromx,fromy,tox,toy){
    var numberCell=$('#number-cell-'+fromx+'-'+fromy);
    numberCell.animate({
        top:setPos(tox,null),
        left:setPos(null,toy)
    },150)
    //var t=tox*1.2+0.2+'rem'
    //var l=toy*1.2+0.2+'rem'
    var toNumberCell=$('#number-cell-'+tox+'-'+toy);
    toNumberCell.css('transform','scale(1.3)')
}
