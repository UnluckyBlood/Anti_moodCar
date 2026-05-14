package com.example.automooden

data class CarResponse(

    val found: Boolean,

    val plate_number: String,

    val owner_name: String,

    val rating: Double,

    val votes: Int,

    val warning: String?
)