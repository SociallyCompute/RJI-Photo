//dynamically build a single row
function addRow(idx){
    $("#contentBox").html("");
    var rowDiv = document.createElement('div');
    var item1 = document.createElement('div');
    var item2 = document.createElement('div');
    var item3 = document.createElement('div');
    var item4 = document.createElement('div');
    var img1 = document.createElement('div');
    var img2 = document.createElement('div');
    var img3 = document.createElement('div');
    var img4 = document.createElement('div');
    var text1 = document.createElement('div');
    var text2 = document.createElement('div');
    var text3 = document.createElement('div');
    var text4 = document.createElement('div');

    rowDiv.className = 'row';
    item1.className = 'item';
    item2.className = 'item';
    item3.className = 'item';
    item4.className = 'item';
    img1.className = 'image';
    img2.className = 'image';
    img3.className = 'image';
    img4.className = 'image';
    text1.className = 'text';
    text2.className = 'text';
    text3.className = 'text';
    text4.className = 'text';

    item1.appendChild(img1);
    item1.appendChild(text1);
    item2.appendChild(img2);
    item2.appendChild(text2);
    item3.appendChild(img3);
    item3.appendChild(text3);
    item4.appendChild(img4);
    item4.appendChild(text4);

    rowDiv.appendChild(item1);
    rowDiv.appendChild(item2);
    rowDiv.appendChild(item3);
    rowDiv.appendChild(item4);

    $(".rows").append(rowDiv);
}

//build the prev/next button row
function addButtons(){
    var buttonDiv = document.createElement('div');
    var nextButton = document.createElement('button');
    var prevButton = document.createElement('button');

    nextButton.className = 'button';
    prevButton.className = 'button';

    nextButton.onclick = nextImages();
    prevButton.onclick = prevImages();

    buttonDiv.appendChild(prevButton);
    buttonDiv.appendChild(nextButton);

    $(".rows").after(buttonDiv);
}

//build the full page, rows and buttons
function buildPage(){
    var i;
    for(i=0;i < 10; i++){
        addRow(i);
    }
    addButtons();
}