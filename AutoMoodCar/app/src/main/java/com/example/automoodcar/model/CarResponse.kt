package com.example.automoodcar.model

data class CarResponse(
    val plate_number: String,
    val owner_name: String,
    val rating: Double,
    val votes: Int,
    val warning: String,
    val found: Boolean
)