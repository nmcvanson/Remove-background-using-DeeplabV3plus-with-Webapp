$(document).ready(function () {
    $('#input_file').on('change', function () {
        file = $('#input_file').prop('files')[0];
        var form_data = new FormData();
        form_data.append('file', file);

        $.ajax({
            url: '/api/process',
            type: "post",
            data: form_data,
            enctype: 'multipart/form-data',
            contentType: false,
            processData: false,
            cache: false,
            beforeSend: function () {
                $(".overlay").show()
            },
        }).done(function (jsondata, textStatus, jqXHR) {
            payload = jsondata
            $('#img-product').attr('src', payload.img)

            $(".overlay").hide()
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $(".overlay").hide()
        });

    })
})